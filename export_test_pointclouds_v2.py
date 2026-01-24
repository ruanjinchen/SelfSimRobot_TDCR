#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_test_pointclouds.py

一个用于 SelfSimRobot_TDCR (你改成 multi-view 的版本) 的 demo：

- 读取你训练用的 .npz 数据集（里面有 angles / rays_o / rays_d / near / far / stems）
- 加载训练好的 best_model.pt
- 根据 test 集（优先用 GT 点云目录里的文件名列表作为 test 列表；否则用 npz 或 split_file）
  对每个样本：
    - 输入 motor/actuator 向量（angles）
    - 用模型在多视角 rays 上采样，生成预测点云（pred）
    - 把对应 GT 点云复制/软链接到 out_dir/gt

输出目录结构：
  out_dir/
    gt/
      000123.npy(or .ply)
    pred/
      000123.npy(or .ply)

建议你把这个脚本放到 SelfSimRobot_TDCR-main/ 同级目录运行（保证能 import model/func）。

示例：
  python export_test_pointclouds.py \
    --npz /data/.../sim_2m_no_base.npz \
    --ckpt /path/to/sim_2m_no_base/best_model/best_model.pt \
    --gt_dir /path/to/gt_pointclouds_test \
    --out_dir /path/to/export_pc \
    --views all \
    --n_samples 64 \
    --score_mode density \
    --alpha_thresh 0.08 \
    --point_mode max \
    --voxel 0.002 \
    --max_points 200000

如果你想只输入一个 motor 向量做单次导出：
  python export_test_pointclouds.py \
    --npz /data/.../sim_2m_no_base.npz \
    --ckpt /path/to/best_model.pt \
    --out_dir /tmp/one_case \
    --angle "0.1,0.2,0.3,..." \
    --name custom001

注意：
- 如果同时提供了 --split_file，则按 --split/--split_file 选择样本；gt_dir 仅用于复制/链接 GT。
- 如果没有提供 --split_file 且 npz 内也没有 split indices，则会用 GT 目录里的文件名作为样本列表（最不容易和 GT 对不上）。
- pred 点云是从 (多视角) rays 上采样后阈值化得到的“伪表面/占据点”。如果你需要更标准的体素/TSDF/mesh，再另做重建。
"""

from __future__ import annotations

import argparse
import os
import shutil
import math
import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch

# 这些 import 假设脚本和 model.py / func.py 在同一目录（仓库根目录）
from model import FBV_SM, PositionalEncoder
from func import model_forward


def _parse_views(s: str, num_views: int) -> List[int]:
    s = (s or "all").strip().lower()
    if s in ("all", "*", "-1"):
        return list(range(num_views))
    # e.g. "0,2,3"
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vid = int(part)
        if vid < 0 or vid >= num_views:
            raise ValueError(f"view id {vid} out of range [0,{num_views-1}]")
        out.append(vid)
    if len(out) == 0:
        raise ValueError("--views parsed to empty list")
    return out


def _parse_angle_list(s: str) -> np.ndarray:
    # "0.1,0.2,0.3" -> (DOF,)
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if len(parts) == 0:
        raise ValueError("--angle is empty")
    return np.asarray([float(x) for x in parts], dtype=np.float32)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_symlink_or_copy(src: Path, dst: Path, *, force_copy: bool = False) -> None:
    """优先创建软链接，失败则 copy。"""
    if dst.exists() or dst.is_symlink():
        return
    try:
        if not force_copy:
            os.symlink(str(src), str(dst))
            return
    except Exception:
        pass
    shutil.copy2(str(src), str(dst))


def _write_ply_ascii(points_xyz: np.ndarray, ply_path: Path) -> None:
    """写一个最简单的 ASCII PLY（只有 xyz，没有颜色）。"""
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points_xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in points_xyz:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def _voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """一个不依赖 open3d 的简单 voxel grid downsample：每个体素保留一个点。"""
    if voxel is None or voxel <= 0:
        return points
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] == 0:
        return pts
    # 量化到体素坐标
    q = np.floor(pts / float(voxel)).astype(np.int64)
    # np.unique 按行去重，返回保留的第一个索引
    _, idx = np.unique(q, axis=0, return_index=True)
    return pts[np.sort(idx)]


def _subsample_rays_by_stride(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    H: int,
    W: int,
    stride: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """把 (H*W,3) 的 rays 按 stride 下采样到 ((H/stride)*(W/stride),3)。"""
    if stride <= 1:
        return rays_o, rays_d
    ro = rays_o.reshape(H, W, 3)[::stride, ::stride].reshape(-1, 3)
    rd = rays_d.reshape(H, W, 3)[::stride, ::stride].reshape(-1, 3)
    return ro, rd


def _resolve_indices_from_gt_dir(
    stems_all: np.ndarray,
    gt_dir: Path,
    gt_ext: str,
) -> Tuple[List[int], List[str]]:
    """用 GT 目录里的文件名（stem）作为 test 列表，并映射回 npz 索引。"""
    gt_ext = gt_ext if gt_ext.startswith(".") else ("." + gt_ext)
    files = sorted(gt_dir.glob(f"*{gt_ext}"))
    if len(files) == 0:
        raise FileNotFoundError(f"No GT files with ext={gt_ext} found in: {gt_dir}")

    stem_to_idx = {str(s): int(i) for i, s in enumerate(stems_all.tolist())}

    idxs: List[int] = []
    stems: List[str] = []
    missing: List[str] = []
    for fp in files:
        st = fp.stem
        if st in stem_to_idx:
            idxs.append(stem_to_idx[st])
            stems.append(st)
        else:
            missing.append(st)

    if len(missing) > 0:
        print(f"[WARN] {len(missing)} GT stems are not found in npz.stems, e.g. {missing[:10]}")

    if len(idxs) == 0:
        raise RuntimeError("After matching GT stems, got 0 valid samples. Check naming / stems.")

    return idxs, stems


def _resolve_indices_from_split(
    data: np.lib.npyio.NpzFile,
    split_file: Optional[Path],
    split_name: str,
    seed: int,
    default_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Tuple[np.ndarray, str]:
    """解析 train/val/test split。

    优先级：
    1) npz 内有 train_idx/val_idx/test_idx
    2) split_file 内有 train_idx/val_idx/test_idx
    3) fallback：按 seed 做一个 80/10/10 的随机 split（并提示用户最好给 split_file）

    返回：indices, source_desc
    """
    split_name = split_name.lower().strip()
    if split_name not in ("train", "val", "valid", "test"):
        raise ValueError("--split must be one of train/val/test")
    split_key = "val" if split_name in ("val", "valid") else split_name

    # 1) from npz keys
    keys = set(getattr(data, "files", []))
    if {"train_idx", "val_idx", "test_idx"}.issubset(keys):
        idx = np.asarray(data[f"{split_key}_idx"], dtype=np.int64)
        return idx, "npz:{train/val/test}_idx"

    # 2) from split_file
    if split_file is not None and split_file.exists():
        sp = np.load(str(split_file), allow_pickle=True)
        sp_keys = set(getattr(sp, "files", []))
        # 兼容 train.py 保存的 split_indices.npz（只有 train_idx / val_idx）
        if split_key == "test" and ("test_idx" not in sp_keys):
            # 优先：如果 split_file 里有 train_idx 和 val_idx，则尝试用补集当 test
            if ("train_idx" in sp_keys) and ("val_idx" in sp_keys):
                N = int(data["angles"].shape[0])
                used = np.zeros((N,), dtype=np.bool_)
                used[np.asarray(sp["train_idx"], dtype=np.int64)] = True
                used[np.asarray(sp["val_idx"], dtype=np.int64)] = True
                comp = np.nonzero(~used)[0].astype(np.int64)
                if comp.size > 0:
                    return comp, f"split_file:{split_file.name} (derived test=complement of train/val)"
            # 否则：退化成用 val 当 test（更适合你现在“没有单独 test”的训练流程）
            if "val_idx" in sp_keys:
                idx = np.asarray(sp["val_idx"], dtype=np.int64)
                return idx, f"split_file:{split_file.name} (fallback test<-val_idx)"
        if f"{split_key}_idx" in sp_keys:
            idx = np.asarray(sp[f"{split_key}_idx"], dtype=np.int64)
            return idx, f"split_file:{split_file.name}"

    # 3) fallback random split
    N = int(data["angles"].shape[0])
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    r_tr, r_val, r_te = default_ratio
    n_tr = int(round(N * r_tr))
    n_val = int(round(N * r_val))
    n_te = N - n_tr - n_val
    train_idx = perm[:n_tr]
    val_idx = perm[n_tr:n_tr + n_val]
    test_idx = perm[n_tr + n_val:]

    if split_key == "train":
        return train_idx, "fallback_random_split(train)"
    if split_key == "val":
        return val_idx, "fallback_random_split(val)"
    return test_idx, "fallback_random_split(test)"


def predict_pointcloud_for_angle(
    angle: torch.Tensor,
    model: torch.nn.Module,
    rays_o_all: torch.Tensor,
    rays_d_all: torch.Tensor,
    nears: np.ndarray,
    fars: np.ndarray,
    H: int,
    W: int,
    view_ids: List[int],
    *,
    n_samples: int,
    chunksize: int,
    score_mode: str,
    alpha_thresh: float,
    point_mode: str,
    ray_stride: int,
    max_points_per_view: int,
    device: torch.device,
) -> np.ndarray:
    """给定一个 motor angle，输出预测点云 (N,3) numpy。"""

    score_mode = score_mode.lower().strip()
    point_mode = point_mode.lower().strip()
    if score_mode not in ("density", "vis", "vis_density"):
        raise ValueError("--score_mode must be density|vis|vis_density")
    if point_mode not in ("all", "max"):
        raise ValueError("--point_mode must be all|max")

    pts_list: List[np.ndarray] = []

    with torch.no_grad():
        for vid in view_ids:
            ro = rays_o_all[vid]
            rd = rays_d_all[vid]
            ro, rd = _subsample_rays_by_stride(ro, rd, H, W, ray_stride)

            near_v = float(nears[vid]) if np.ndim(nears) > 0 else float(nears)
            far_v = float(fars[vid]) if np.ndim(fars) > 0 else float(fars)

            # output_flag=3: 返回 (rgb_map, query_points, density(alpha), visibility(raw0))
            rgb_map, query_points, density, visibility = model_forward(
                ro, rd,
                near_v, far_v,
                model,
                arm_angle=angle,
                DOF=int(angle.shape[0]),
                chunksize=chunksize,
                n_samples=n_samples,
                output_flag=3,
            )

            if score_mode == "density":
                score = density
            elif score_mode == "vis":
                score = torch.sigmoid(visibility)
            else:  # vis_density
                score = density * torch.sigmoid(visibility)

            if point_mode == "all":
                mask = score > float(alpha_thresh)
                pts = query_points[mask]  # (N,3)
            else:
                # 每条 ray 只取 score 最大的那个点（更像“表面点”）
                max_score, max_idx = torch.max(score, dim=1)  # (N_rays,), (N_rays,)
                hit = max_score > float(alpha_thresh)
                if torch.any(hit):
                    ar = torch.arange(query_points.shape[0], device=query_points.device)
                    pts = query_points[ar, max_idx]  # (N_rays,3)
                    pts = pts[hit]
                else:
                    pts = query_points.new_zeros((0, 3))

            if pts.numel() == 0:
                continue

            pts_np = pts.detach().cpu().numpy().astype(np.float32)

            if (max_points_per_view is not None) and (max_points_per_view > 0) and (pts_np.shape[0] > max_points_per_view):
                sel = np.random.choice(pts_np.shape[0], size=max_points_per_view, replace=False)
                pts_np = pts_np[sel]

            pts_list.append(pts_np)

    if len(pts_list) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    return np.concatenate(pts_list, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True, help="Path to dataset .npz")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to trained model checkpoint (.pt), e.g. .../best_model.pt")

    ap.add_argument("--out_dir", type=str, required=True, help="Output directory root")
    ap.add_argument("--gt_dir", type=str, default=None, help="GT point cloud directory (used to define test list + linking/copying)")
    ap.add_argument("--gt_ext", type=str, default=".npy", help="GT file extension (.npy/.ply/.xyz ...). Default .npy")
    ap.add_argument("--pred_ext", type=str, default=None, help="Prediction output extension (.npy/.ply). If omitted: use --gt_ext when --gt_dir is set, else .npy")

    ap.add_argument("--split", type=str, default="test", help="Which split to export if gt_dir not provided: train/val/test")
    ap.add_argument("--split_file", type=str, default=None,
                    help="Optional split file (.npz) containing train_idx/val_idx/(test_idx). Can use train.py's split_indices.npz")

    ap.add_argument("--views", type=str, default="all", help="Which camera views to use: 'all' or '0,1,2'")
    ap.add_argument("--n_samples", type=int, default=64, help="Samples per ray")
    ap.add_argument("--chunksize", type=int, default=2**20, help="Chunk size for MLP inference")

    ap.add_argument("--score_mode", type=str, default="density", choices=["density", "vis", "vis_density"],
                    help="How to score occupancy at sampled points")
    ap.add_argument("--alpha_thresh", type=float, default=0.08, help="Threshold for selecting occupied points")
    ap.add_argument("--point_mode", type=str, default="max", choices=["max", "all"],
                    help="max: 1 point per ray (argmax). all: keep all points above threshold")

    ap.add_argument("--ray_stride", type=int, default=1, help="Subsample rays by pixel stride (1=use all rays, 2=every 2 pixels)")
    ap.add_argument("--max_points_per_view", type=int, default=0, help="Optional cap per view after thresholding (0=disabled)")
    ap.add_argument("--voxel", type=float, default=0.0, help="Voxel downsample size in world units (0=disable)")
    ap.add_argument("--max_points", type=int, default=0, help="Final cap on total points (0=disable)")

    ap.add_argument("--seed", type=int, default=1, help="Random seed (for subsampling)")
    ap.add_argument("--copy_gt", action="store_true", help="Copy GT files instead of symlink")

    # 单次输入 motor 向量
    ap.add_argument("--angle", type=str, default=None, help="Comma-separated motor values. If set, export only one case.")
    ap.add_argument("--name", type=str, default="custom", help="Name/stem used when --angle is provided")

    # model config
    ap.add_argument("--d_filter", type=int, default=128, help="Hidden width (must match training)")
    ap.add_argument("--n_freqs", type=int, default=10, help="Positional encoding freqs (must match training)")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    npz_path = Path(args.npz).expanduser()
    ckpt_path = Path(args.ckpt).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    gt_dir = Path(args.gt_dir).expanduser() if args.gt_dir else None
    split_file = Path(args.split_file).expanduser() if args.split_file else None

    if not npz_path.exists():
        raise FileNotFoundError(f"npz not found: {npz_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    data = np.load(str(npz_path), allow_pickle=True)
    if "angles" not in data.files or "rays_o" not in data.files or "rays_d" not in data.files:
        raise ValueError(f"npz missing required keys. keys={data.files}")

    angles_np = data["angles"].astype(np.float32)
    DOF = int(angles_np.shape[1])

    rays_o_np = data["rays_o"].astype(np.float32)  # (V, H*W, 3)
    rays_d_np = data["rays_d"].astype(np.float32)
    num_views = int(rays_o_np.shape[0])

    # infer H,W
    if "images" in data.files:
        H = int(data["images"].shape[2])
        W = int(data["images"].shape[3])
    else:
        HW = int(rays_o_np.shape[1])
        side = int(round(math.sqrt(HW)))
        if side * side != HW:
            raise ValueError(f"Cannot infer H/W from rays_o length={HW}. Provide images in npz.")
        H = W = side

    nears = data["near"].astype(np.float32) if "near" in data.files else np.array([0.1] * num_views, dtype=np.float32)
    fars = data["far"].astype(np.float32) if "far" in data.files else np.array([2.0] * num_views, dtype=np.float32)

    stems_all = data["stems"].astype(str) if "stems" in data.files else np.asarray([f"{i:06d}" for i in range(len(angles_np))])

    # move rays to torch (keep on GPU for speed)
    rays_o_all = torch.from_numpy(rays_o_np).to(device)
    rays_d_all = torch.from_numpy(rays_d_np).to(device)

    view_ids = _parse_views(args.views, num_views=num_views)
    print(f"[INFO] dataset: N={len(angles_np)}  DOF={DOF}  views={num_views}  HxW={H}x{W}")
    print(f"[INFO] using views: {view_ids}")

    # build model (must match training)
    d_input = DOF + 3
    encoder = PositionalEncoder(d_input=d_input, n_freqs=int(args.n_freqs), log_space=True)
    model = FBV_SM(encoder=encoder, d_input=d_input, d_filter=int(args.d_filter), output_size=2).to(device)

    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[INFO] loaded ckpt: {ckpt_path}")

    # output folders
    pred_dir = out_dir / "pred"
    gt_out_dir = out_dir / "gt"
    _ensure_dir(pred_dir)
    _ensure_dir(gt_out_dir)

    gt_ext = args.gt_ext if args.gt_ext.startswith(".") else ("." + args.gt_ext)
    if args.pred_ext is None:
        # 默认：如果给了 gt_dir，则 pred 和 gt 用同一种后缀；否则用 .npy
        pred_ext = gt_ext if (gt_dir is not None) else ".npy"
    else:
        pred_ext = args.pred_ext if args.pred_ext.startswith(".") else ("." + args.pred_ext)

    # --------------------------------------------------
    # Decide which cases to export
    # --------------------------------------------------
    if args.angle is not None:
        # single custom case
        ang = _parse_angle_list(args.angle)
        if ang.shape[0] != DOF:
            raise ValueError(f"--angle length {ang.shape[0]} != DOF {DOF}")
        case_indices = [None]  # type: ignore
        case_stems = [str(args.name)]
        case_angles = [ang]
        print(f"[INFO] single case: {case_stems[0]}  angle.shape={ang.shape}")
    else:
        # batch cases
        # 选择样本列表的优先级：
        # - 如果提供了 split_file（或 npz 内自带 split indices），则按 --split 选择样本；
        #   gt_dir 仅用于读取/复制 GT 点云。
        # - 否则如果提供了 gt_dir，则用 gt_dir 里的文件名作为样本列表（更不容易和 GT 不对齐）。
        has_split_keys = any(k in data.files for k in ("train_idx", "val_idx", "test_idx"))
        if (split_file is None) and (not has_split_keys) and (gt_dir is not None):
            if not gt_dir.exists():
                raise FileNotFoundError(f"gt_dir not found: {gt_dir}")
            idxs, stems = _resolve_indices_from_gt_dir(stems_all, gt_dir, gt_ext)
            case_indices = idxs
            case_stems = stems
            print(f"[INFO] cases from gt_dir: {len(case_indices)}")
        else:
            idx, src = _resolve_indices_from_split(data, split_file, args.split, args.seed)
            case_indices = idx.tolist()
            case_stems = [str(stems_all[i]) for i in case_indices]
            print(f"[INFO] cases from split='{args.split}' ({src}): {len(case_indices)}")
        case_angles = [angles_np[i] for i in case_indices]  # type: ignore

    # --------------------------------------------------
    # Export loop
    # --------------------------------------------------
    total = len(case_stems)
    for k, (stem, ang_np) in enumerate(zip(case_stems, case_angles)):
        ang_t = torch.from_numpy(np.asarray(ang_np, dtype=np.float32)).to(device)

        pts = predict_pointcloud_for_angle(
            angle=ang_t,
            model=model,
            rays_o_all=rays_o_all,
            rays_d_all=rays_d_all,
            nears=nears,
            fars=fars,
            H=H,
            W=W,
            view_ids=view_ids,
            n_samples=int(args.n_samples),
            chunksize=int(args.chunksize),
            score_mode=args.score_mode,
            alpha_thresh=float(args.alpha_thresh),
            point_mode=args.point_mode,
            ray_stride=int(args.ray_stride),
            max_points_per_view=(int(args.max_points_per_view) if int(args.max_points_per_view) > 0 else 0),
            device=device,
        )

        # merge / downsample / cap
        if args.voxel and float(args.voxel) > 0:
            pts = _voxel_downsample(pts, float(args.voxel))
        if args.max_points and int(args.max_points) > 0 and pts.shape[0] > int(args.max_points):
            sel = np.random.choice(pts.shape[0], size=int(args.max_points), replace=False)
            pts = pts[sel]

        # save pred
        pred_path = pred_dir / f"{stem}{pred_ext}"
        if pred_ext.lower() == ".npy":
            np.save(str(pred_path), pts.astype(np.float32))
        elif pred_ext.lower() == ".ply":
            _write_ply_ascii(pts, pred_path)
        else:
            # 兜底：写 xyz 文本
            np.savetxt(str(pred_path), pts.astype(np.float32), fmt="%.6f")

        # link/copy gt
        if (gt_dir is not None) and (args.angle is None):
            gt_src = gt_dir / f"{stem}{gt_ext}"
            if gt_src.exists():
                gt_dst = gt_out_dir / gt_src.name
                _safe_symlink_or_copy(gt_src, gt_dst, force_copy=bool(args.copy_gt))
            else:
                print(f"[WARN] missing GT file: {gt_src}")

        if (k % 10 == 0) or (k == total - 1):
            print(f"[PROGRESS] {k+1}/{total}  stem={stem}  pred_points={pts.shape[0]}")

    print(f"[DONE] exported to: {out_dir}")


if __name__ == "__main__":
    main()
'''

python export_test_pointclouds_v2.py \
  --npz /data/yxk/K-data/K/fllm-sm/sim/sim_2m_no_base.npz \
  --ckpt /data/yxk/K-data/K/nllm/sim_2m_no_base/best_model/best_model.pt \
  --gt_dir /data/yxk/K-data/K/fllm-sm/sim/2m_no_base/pointcloud \
  --gt_ext .ply \
  --out_dir demo_out/2m_no_base \
  --split test \
  --split_file /data/yxk/K-data/K/nllm/sim_2m_no_base/split_indices.npz \
  --views all \
  --n_samples 64 \
  --score_mode density \
  --alpha_thresh 0.08 \
  --point_mode max \
  --voxel 0.002 \
  --max_points 200000

'''