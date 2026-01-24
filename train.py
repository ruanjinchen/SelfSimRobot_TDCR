# Our implementation is based on the NeRF publicly available code from https://github.com/krrish94/nerf-pytorch/ and
# https://github.com/bmild/nerf
import random
import argparse
from pathlib import Path
import os
import sys
import time
import json
import atexit
import signal
import traceback
import datetime
import platform
from typing import Any, Dict, Optional

from model import FBV_SM, PositionalEncoder
from func import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("train,", device)


# ---------------------------------------------------------------------------
# Run metadata / timing helpers
# Writes: <log_dir>/run_info.json
# ---------------------------------------------------------------------------

RUN_INFO_PATH: Optional[Path] = None
RUN_INFO: Dict[str, Any] = {}
RUN_START_TIME: Optional[float] = None
_ORIG_EXCEPTHOOK = sys.excepthook


def _now_local_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()


def _now_utc_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _duration_dict(seconds: float) -> Dict[str, Any]:
    seconds = float(max(0.0, seconds))
    return {
        "seconds": seconds,
        "minutes": seconds / 60.0,
        "hours": seconds / 3600.0,
        "days": seconds / 86400.0,
        "human": f"{seconds/3600.0:.6f} h ({seconds/60.0:.2f} min, {seconds/86400.0:.6f} d)",
    }


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _get_gpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if torch.cuda.is_available():
        try:
            cur = int(torch.cuda.current_device())
            props = torch.cuda.get_device_properties(cur)
            info.update({
                "current_device_index": cur,
                "device_name": torch.cuda.get_device_name(cur),
                "device_count": int(torch.cuda.device_count()),
                "capability": [int(props.major), int(props.minor)],
                "total_memory_bytes": int(props.total_memory),
                "total_memory_GiB": float(props.total_memory / (1024 ** 3)),
                "multi_processor_count": int(props.multi_processor_count),
            })
        except Exception as e:
            info["error"] = f"Failed to query torch.cuda device properties: {e}"
    return info


def _finalize_run_info(final_status: str, *, error: str | None = None, tb: str | None = None, signal_name: str | None = None) -> None:
    global RUN_INFO
    if RUN_INFO_PATH is None:
        return

    RUN_INFO["status"] = final_status
    RUN_INFO["end_time_local"] = _now_local_iso()
    RUN_INFO["end_time_utc"] = _now_utc_iso()

    if RUN_START_TIME is not None:
        RUN_INFO["duration"] = _duration_dict(time.time() - RUN_START_TIME)

    if signal_name is not None:
        RUN_INFO["signal"] = signal_name
    if error is not None:
        RUN_INFO["error"] = error
    if tb is not None:
        RUN_INFO["traceback"] = tb

    # Refresh GPU info at the end too (useful if the run is long)
    RUN_INFO["gpu"] = _get_gpu_info()

    try:
        _atomic_write_json(RUN_INFO_PATH, RUN_INFO)
    except Exception as e:
        print(f"[WARN] Failed to write run_info.json at finalize: {e}")


def _setup_runinfo_hooks() -> None:
    # Unhandled exception hook -> status=failed/interrupted
    def _excepthook(exctype, value, tb):
        if RUN_INFO_PATH is not None:
            status = "interrupted" if exctype is KeyboardInterrupt else "failed"
            tb_str = "".join(traceback.format_exception(exctype, value, tb))
            _finalize_run_info(status, error=str(value), tb=tb_str)
        _ORIG_EXCEPTHOOK(exctype, value, tb)

    sys.excepthook = _excepthook

    # Normal interpreter exit -> status=completed (if still running)
    def _atexit_hook():
        if RUN_INFO_PATH is None:
            return
        if RUN_INFO.get("status") == "running":
            _finalize_run_info("completed")

    atexit.register(_atexit_hook)

    # Handle SIGTERM nicely (common in cluster schedulers)
    def _sigterm_handler(signum, frame):
        if RUN_INFO_PATH is not None and RUN_INFO.get("status") == "running":
            _finalize_run_info("terminated", error=f"Received signal {signum}", signal_name="SIGTERM")
        raise SystemExit(128 + signum)

    try:
        signal.signal(signal.SIGTERM, _sigterm_handler)
    except Exception:
        pass


def crop_center(
        img: torch.Tensor,
        frac: float = 0.5
) -> torch.Tensor:
    r"""
  Crop center square from image.
  """
    h_offset = round(img.shape[0] * (frac / 2))
    w_offset = round(img.shape[1] * (frac / 2))
    return img[h_offset:-h_offset, w_offset:-w_offset]


def init_models(d_input, d_filter, pretrained_model_pth=None, lr=5e-4, output_size=2,FLAG_PositionalEncoder = False):

    if FLAG_PositionalEncoder:
        encoder = PositionalEncoder(d_input, n_freqs=10, log_space=True)

        model = FBV_SM(encoder = encoder,
                       d_input=d_input,
                       d_filter=d_filter,
                       output_size=output_size)

    else:
        # Models
        model = FBV_SM(d_input=d_input,
                       d_filter=d_filter,
                       output_size=output_size)
    model.to(device)
    # Pretrained Model
    if pretrained_model_pth != None:
        model.load_state_dict(torch.load(pretrained_model_pth + "best_model.pt", map_location=torch.device(device)))
    # Optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer



def train(model, optimizer):

    loss_v_last = np.inf
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=20, verbose=True
    )
    patience = 0
    min_loss = np.inf

    # Multi-view tensors:
    #   training_img:   (N_train, V, H, W)
    #   training_angles:(N_train, DOF)
    height, width = training_img.shape[2:4]
    num_views = training_img.shape[1]

    for i in trange(n_iters):
        model.train()

        # ---- sample a robot state ----
        target_state_idx = np.random.randint(training_img.shape[0])
        angle = training_angles[target_state_idx]

        # ---- choose which views to use this step ----
        if (views_per_step is None) or (views_per_step <= 0) or (views_per_step >= num_views):
            view_ids = list(range(num_views))
        else:
            view_ids = np.random.choice(num_views, size=views_per_step, replace=False).tolist()

        # ---- forward & loss over selected views ----
        view_losses = []
        for vid in view_ids:
            target_img = training_img[target_state_idx, vid]  # (H,W)
            rays_o = rays_o_all[vid]
            rays_d = rays_d_all[vid]
            near_v = float(nears[vid]) if np.ndim(nears) > 0 else float(nears)
            far_v = float(fars[vid]) if np.ndim(fars) > 0 else float(fars)

            if center_crop and i < center_crop_iters:
                target_img = crop_center(target_img)
                rays_o, rays_d = crop_rays_center(rays_o, rays_d, height, width, frac=0.5)

            target_flat = target_img.reshape(-1).to(device)

            outputs = model_forward(
                rays_o, rays_d,
                near_v, far_v,
                model,
                arm_angle=angle,
                DOF=DOF,
                chunksize=chunksize,
                output_flag=different_arch
            )

            pred = outputs['rgb_map']
            # view_losses.append(torch.nn.functional.mse_loss(pred, target_flat))
            # 改为 前景加权 MSE
            w_pos = 30.0
            w = 1.0 + w_pos * target_flat   # target_flat 是 0/1
            view_losses.append(((pred - target_flat) ** 2 * w).mean())


        loss = torch.stack(view_losses).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train = loss.item()

        # ---- validation ----
        if i % display_rate == 0:
            model.eval()
            valid_epoch_loss = []
            valid_image = []

            # small, fast validation subset (otherwise multi-view validation is very expensive)
            n_eval_states = min(valid_amount_eval, len(testing_angles))
            eval_state_ids = np.random.choice(len(testing_angles), size=n_eval_states, replace=False)

            # for validation, sample a few views too (or all if you want)
            if (valid_views_eval is None) or (valid_views_eval <= 0) or (valid_views_eval >= num_views):
                eval_view_ids = list(range(num_views))
            else:
                eval_view_ids = np.random.choice(num_views, size=valid_views_eval, replace=False).tolist()

            vis_view = 0  # fixed view id for visualization images
            with torch.no_grad():
                for k, v_i in enumerate(eval_state_ids):
                    angle = testing_angles[v_i]
                    per_view_losses = []

                    for vid in eval_view_ids:
                        img_label = testing_img[v_i, vid]
                        rays_o = rays_o_all[vid]
                        rays_d = rays_d_all[vid]
                        near_v = float(nears[vid]) if np.ndim(nears) > 0 else float(nears)
                        far_v = float(fars[vid]) if np.ndim(fars) > 0 else float(fars)

                        outputs = model_forward(
                            rays_o, rays_d,
                            near_v, far_v,
                            model,
                            arm_angle=angle,
                            DOF=DOF,
                            chunksize=chunksize,
                            output_flag=different_arch
                        )
                        pred = outputs['rgb_map']
                        label_flat = img_label.reshape(-1).to(device)
                        per_view_losses.append(torch.nn.functional.mse_loss(pred, label_flat))

                        # Visualization: only save predictions from one fixed view
                        if (k < max_pic_save) and (vid == vis_view):
                            np_image = pred.reshape([height, width, 1]).detach().cpu().numpy()
                            valid_image.append(np_image)

                    valid_epoch_loss.append(torch.stack(per_view_losses).mean().item())

            loss_valid = float(np.mean(valid_epoch_loss))
            print("Val Loss:", loss_valid, 'patience', patience)
            scheduler.step(loss_valid)

            # ---- update run_info.json (progress snapshot) ----
            if RUN_INFO_PATH is not None and RUN_START_TIME is not None:
                try:
                    RUN_INFO["progress"] = {
                        "iter": int(i),
                        "train_loss": float(loss_train),
                        "val_loss": float(loss_valid),
                        "patience": int(patience),
                        "min_val_loss": float(min_loss),
                        "lr": float(optimizer.param_groups[0].get("lr", 0.0)),
                        "timestamp_local": _now_local_iso(),
                        "elapsed": _duration_dict(time.time() - RUN_START_TIME),
                        "views_per_step": int(len(view_ids)),
                        "eval_views": int(len(eval_view_ids)),
                        "eval_states": int(n_eval_states),
                    }
                    _atomic_write_json(RUN_INFO_PATH, RUN_INFO)
                except Exception as e:
                    print(f"[WARN] Failed to update run_info.json: {e}")

            # Save validation image strip
            if len(valid_image) > 0:
                np_image_combine = np.hstack(valid_image)
                np_image_combine = np.dstack((np_image_combine, np_image_combine, np_image_combine))
                np_image_combine = np.clip(np_image_combine, 0, 1)
                matplotlib.image.imsave(LOG_PATH + '/image/' + 'latest.png', np_image_combine)
                if Flag_save_image_during_training:
                    matplotlib.image.imsave(LOG_PATH + '/image/' + '%d.png' % i, np_image_combine)

            record_file_train.write(str(loss_train) + "\n")
            record_file_val.write(str(loss_valid) + "\n")
            torch.save(model.state_dict(), LOG_PATH + '/best_model/model_epoch%d.pt' % i)

            if min_loss > loss_valid:
                # record the best image and model
                min_loss = loss_valid
                if len(valid_image) > 0:
                    matplotlib.image.imsave(LOG_PATH + '/image/' + 'best.png', np_image_combine)
                torch.save(model.state_dict(), LOG_PATH + '/best_model/best_model.pt')
                patience = 0
            elif loss_valid == loss_v_last:
                print("restart")
                return False
            else:
                patience += 1

            loss_v_last = loss_valid

        if patience > Patience_threshold:
            break

    return True


if __name__ == "__main__":

    ap = argparse.ArgumentParser(
        description="Train the TDCR-adapted SelfSimRobot (multi-view NeRF-style) model on an NPZ dataset."
    )
    ap.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the .npz dataset (expects keys: images, angles, rays_o, rays_d, near, far).",
    )
    ap.add_argument("--seed", type=int, default=1, help="Random seed (python/numpy/torch).")
    ap.add_argument(
        "--select",
        type=int,
        default=10000,
        help="How many samples to use from the dataset (<=N).",
    )
    ap.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train split ratio. Remaining samples are used as validation.",
    )
    ap.add_argument(
        "--views_per_step",
        type=int,
        default=2,
        help="How many camera views to use per SGD step (<=V).",
    )
    ap.add_argument(
        "--valid_amount_eval",
        type=int,
        default=32,
        help="How many robot states to sample during each validation (for speed).",
    )
    ap.add_argument(
        "--valid_views_eval",
        type=int,
        default=2,
        help="How many camera views to sample during each validation (for speed).",
    )
    ap.add_argument(
        "--log_root",
        type=str,
        default="train_log",
        help="Root folder to store training logs.",
    )
    ap.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help=(
            "Optional: full path to the exact log directory for this run. "
            "If set, it overrides the auto-generated name under --log_root."
        ),
    )
    ap.add_argument(
        "--run_name",
        type=str,
        default="",
        help=(
            "Optional tag appended to the log directory name (e.g. 'vps4_evalall'). "
            "Useful to avoid overwriting when running multiple experiments with the same seed."
        ),
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "If set, allow writing into an existing log directory. "
            "(Note: this script does not implement true resume; files may be overwritten.)"
        ),
    )
    ap.add_argument(
        "--split_file",
        type=str,
        default=None,
        help=(
            "Optional path to a .npz file storing a fixed data split (train_idx/val_idx). "
            "If provided and the file exists, the split will be reused so that validation never "
            "accidentally pulls training samples. If provided but the file does not exist, the "
            "generated split will be saved there. If omitted, a split file is saved to "
            "<LOG_PATH>/split_indices.npz."
        ),
    )
    args = ap.parse_args()

    sim_real = 'sim'
    arm_ee = 'ee'
    seed_num = int(args.seed)
    robotid = 1
    FLAG_PositionalEncoder= True

    # 0:OM, 1:OneOut, 2: OneOut with distance
    different_arch = 0
    print('different_arch',different_arch)

    np.random.seed(seed_num)
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    select_data_amount = int(args.select)
    # DOF will be inferred from the dataset after loading it.
    # near/far will be loaded from the dataset (per-view) if present.
    Flag_save_image_during_training = True

    if FLAG_PositionalEncoder:
        add_name = 'PE'
    else:
        add_name = 'no_PE'

    tr = float(args.train_ratio)  # training ratio
    pxs = 100  # collected data pixels

    data_path = Path(args.data).expanduser()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    # data = np.load('data/%s_data/%s_data_robo%d(%s).npz'%(sim_real,sim_real,robotid,arm_ee))
    data = np.load(str(data_path))


    # ---- multi-view additions ----
    DOF = int(data["angles"].shape[1])
    num_views = int(data["images"].shape[1])
    
    # Rays are precomputed per-view in the NPZ (recommended, keeps training independent of MuJoCo).
    rays_o_all = torch.from_numpy(data["rays_o"].astype("float32")).to(device)  # (V, H*W, 3)
    rays_d_all = torch.from_numpy(data["rays_d"].astype("float32")).to(device)  # (V, H*W, 3)
    
    # Near/Far can be scalar or per-view. Prefer per-view arrays.
    nears = data["near"].astype("float32") if "near" in data.files else np.array([0.1] * num_views, dtype=np.float32)
    fars  = data["far"].astype("float32") if "far" in data.files  else np.array([2.0] * num_views, dtype=np.float32)
    # data = np.load('data/%s_data/%s_data_robo%d(%s)_cam%d.npz'%(sim_real,sim_real,robotid,arm_ee,cam_dist*1000))
    # data = np.load('data/%s_data/%s_data_robo%d(%s)_cam%d_test.npz'%(sim_real,sim_real,robotid,arm_ee,800)) # 800 test is 1000 ... local data, Jiong
    num_raw_data = len(data["angles"])
    select_data_amount = min(select_data_amount, num_raw_data)

    print("DOF, num_data, robot_id, PE",DOF,select_data_amount,robotid,FLAG_PositionalEncoder)
    LOG_PATH = os.path.join(
        str(args.log_root),
        "%s_id%d_%d(%d)_%s(%s)" % (sim_real, robotid, select_data_amount, seed_num, add_name, arm_ee),
    )
    # Allow full custom log directory (best for running multiple jobs in parallel)
    if args.log_dir is not None and str(args.log_dir).strip() != "":
        LOG_PATH = str(Path(args.log_dir).expanduser())

    # Optional human-readable tag/suffix
    if args.run_name is not None and str(args.run_name).strip() != "":
        LOG_PATH = LOG_PATH + "_" + str(args.run_name).strip()

    if different_arch != 0:
        LOG_PATH += 'diff_out_%d'%different_arch
    # Avoid accidental overwrites when launching multiple runs with identical settings.
    # If the target directory exists and --overwrite is NOT set, auto-increment a suffix.
    log_path_obj = Path(LOG_PATH)
    if log_path_obj.exists() and (not args.overwrite):
        k = 1
        while True:
            cand = Path(f"{LOG_PATH}_run{k}")
            if not cand.exists():
                LOG_PATH = str(cand)
                break
            k += 1
    print("Data Loaded!")
    os.makedirs(LOG_PATH + "/image/", exist_ok=True)
    os.makedirs(LOG_PATH + "/best_model/", exist_ok=True)

    # ---- initialize run_info.json (status=running) ----
    RUN_START_TIME = time.time()
    RUN_INFO_PATH = Path(LOG_PATH) / "run_info.json"
    RUN_INFO = {
        "status": "running",
        "start_time_local": _now_local_iso(),
        "start_time_utc": _now_utc_iso(),
        "end_time_local": None,
        "end_time_utc": None,
        "duration": None,
        "command": " ".join([sys.executable] + sys.argv),
        "argv": list(sys.argv),
        "pid": int(os.getpid()),
        "cwd": os.getcwd(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "torch": {
            "version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": (torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None),
        },
        "gpu": _get_gpu_info(),
        "args": dict(vars(args)),
        "log_path": str(LOG_PATH),
        "data_path": str(data_path),
    }
    _atomic_write_json(RUN_INFO_PATH, RUN_INFO)
    _setup_runinfo_hooks()


    # ----------------------------------------------------------------------
    # Build / load a reproducible train/val split.
    #
    # Why:
    # - Your dataset .npz typically contains only raw arrays (images/angles/...) and
    #   does NOT store any split.
    # - This training script historically made a random split each run, which makes
    #   it hard to later do a demo that *guarantees* you're only sampling from val.
    #
    # Solution:
    # - If --split_file exists: load train_idx/val_idx from it.
    # - Else: generate a split once (using the current --seed/--select/--train_ratio)
    #   and save it to the split file for future reproducibility.
    # ----------------------------------------------------------------------

    max_pic_save = 6

    split_path = Path(args.split_file).expanduser() if args.split_file else (Path(LOG_PATH) / "split_indices.npz")
    split_path.parent.mkdir(parents=True, exist_ok=True)

    def _check_split_indices(train_idx_in, val_idx_in):
        train_idx_arr = np.asarray(train_idx_in, dtype=np.int64).reshape(-1)
        val_idx_arr = np.asarray(val_idx_in, dtype=np.int64).reshape(-1)

        if train_idx_arr.size == 0 or val_idx_arr.size == 0:
            raise ValueError(
                f"Empty split: train={train_idx_arr.size}, val={val_idx_arr.size}. "
                "Check --train_ratio/--select."
            )

        if train_idx_arr.min() < 0 or val_idx_arr.min() < 0:
            raise ValueError("Split indices must be >= 0.")

        if train_idx_arr.max() >= num_raw_data or val_idx_arr.max() >= num_raw_data:
            raise ValueError(
                f"Split indices out of range for this dataset (num_raw_data={num_raw_data})."
            )

        if np.intersect1d(train_idx_arr, val_idx_arr).size != 0:
            raise ValueError("Split has overlap between train and val indices.")

        return train_idx_arr, val_idx_arr

    if split_path.exists():
        split_npz = np.load(str(split_path), allow_pickle=True)
        if ("train_idx" not in split_npz.files) or ("val_idx" not in split_npz.files):
            raise ValueError(
                f"Split file missing train_idx/val_idx: {split_path} (keys={split_npz.files})"
            )
        train_idx, val_idx = _check_split_indices(split_npz["train_idx"], split_npz["val_idx"])
        print(f"[INFO] Loaded split from {split_path} (train={len(train_idx)} val={len(val_idx)})")
    else:
        sample_id = random.sample(range(num_raw_data), select_data_amount)
        split_point = int(select_data_amount * tr)
        train_idx = np.asarray(sample_id[:split_point], dtype=np.int64)
        val_idx = np.asarray(sample_id[split_point:], dtype=np.int64)
        train_idx, val_idx = _check_split_indices(train_idx, val_idx)

        split_dict = {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "seed": np.asarray(seed_num, dtype=np.int64),
            "select": np.asarray(select_data_amount, dtype=np.int64),
            "train_ratio": np.asarray(tr, dtype=np.float32),
        }
        # Optional: store stems too (much easier to debug / reproduce across scripts)
        if "stems" in data.files:
            try:
                stems = data["stems"]
                split_dict["train_stems"] = stems[train_idx]
                split_dict["val_stems"] = stems[val_idx]
            except Exception:
                pass

        np.savez_compressed(str(split_path), **split_dict)
        print(f"[INFO] Saved split to {split_path} (train={len(train_idx)} val={len(val_idx)})")

    # ---- enrich run_info.json with dataset + split summary ----
    if RUN_INFO_PATH is not None:
        try:
            RUN_INFO["dataset"] = {
                "keys": list(getattr(data, "files", [])),
                "num_raw_data": int(num_raw_data),
                "select_data_amount": int(select_data_amount),
                "DOF": int(DOF),
                "num_views": int(num_views),
                "images_shape": list(data["images"].shape) if "images" in data.files else None,
                "angles_shape": list(data["angles"].shape) if "angles" in data.files else None,
                "rays_o_shape": list(data["rays_o"].shape) if "rays_o" in data.files else None,
                "rays_d_shape": list(data["rays_d"].shape) if "rays_d" in data.files else None,
                "near_shape": list(np.asarray(data["near"]).shape) if "near" in data.files else None,
                "far_shape": list(np.asarray(data["far"]).shape) if "far" in data.files else None,
            }
            RUN_INFO["split"] = {
                "split_file": str(split_path),
                "train_size": int(len(train_idx)),
                "val_size": int(len(val_idx)),
                "train_ratio": float(tr),
                "seed": int(seed_num),
            }
            _atomic_write_json(RUN_INFO_PATH, RUN_INFO)
        except Exception as e:
            print(f"[WARN] Failed to write dataset/split summary to run_info.json: {e}")

    # ---- Select a few validation images for visualization ----
    vis_ids = val_idx[: min(max_pic_save, len(val_idx))]
    valid_img_visual = np.hstack(data['images'][vis_ids, 0])
    valid_angle = data['angles'][vis_ids]
    np.savetxt(LOG_PATH+'/image/valid_angle.csv', valid_angle)

    # Repeat the stacked image three times along the depth
    valid_img_visual = np.dstack((valid_img_visual, valid_img_visual, valid_img_visual))
    print("Valid Data Loaded!")

    # Gather as torch tensors
    training_img = torch.from_numpy(data['images'][train_idx].astype('float32'))
    training_angles = torch.from_numpy(data['angles'][train_idx].astype('float32'))

    testing_img = torch.from_numpy(data['images'][val_idx].astype('float32'))
    testing_angles = torch.from_numpy(data['angles'][val_idx].astype('float32'))
    train_amount = len(training_angles)
    valid_amount = len(testing_angles)
    print(valid_amount)

    # Grab rays from sample image
    height, width = training_img.shape[2:4]
    print('IMG (height, width)', (height, width))

    # Encoders
    """arm dof = 2+3; arm dof=3+3"""

    # Stratified sampling
    n_samples = 64  # Number of spatial samples per ray
    perturb = True  # If set, applies noise to sample positions
    inverse_depth = False  # If set, samples points linearly in inverse depth

    # Hierarchical sampling
    n_samples_hierarchical = 64  # Number of samples per ray
    perturb_hierarchical = False  # If set, applies noise to sample positions

    # Training
    n_iters = 400000
    one_image_per_step = True  # One image per gradient step (disables batching)
    chunksize = 2 ** 20  # Modify as needed to fit in GPU memory
    center_crop = False  # Crop the center of image (one_image_per_)   # debug
    center_crop_iters = 200  # Stop cropping center after this many epochs
    display_rate = 1000 #int(select_data_amount*tr)  # Display test output every X epochs

    # Multi-view training controls
    # - If views_per_step >= num_views: use all views for the sampled robot state.
    # - If views_per_step == 1: this degenerates to single-view SGD but still trains on all cameras over time.
    views_per_step = int(args.views_per_step)

    # Validation controls (keep small for speed; multi-view validation is expensive)
    valid_amount_eval = int(args.valid_amount_eval)
    valid_views_eval = int(args.valid_views_eval)

    # Early Stopping
    warmup_iters = 400  # Number of iterations during warmup phase
    warmup_min_fitness = 10.0  # Min val PSNR to continue training at warmup_iters
    n_restarts = 1000  # Number of times to restart if training stalls

    # We bundle the kwargs for various functions to pass all at once.
    kwargs_sample_stratified = {
        'n_samples': n_samples,
        'perturb': perturb,
        'inverse_depth': inverse_depth
    }
    kwargs_sample_hierarchical = {
        'perturb': perturb
    }


    record_file_train = open(LOG_PATH + "/log_train.txt", "w")
    record_file_val = open(LOG_PATH + "/log_val.txt", "w")
    Patience_threshold = 100

    # Save testing gt image for visualization
    matplotlib.image.imsave(LOG_PATH + '/image/' + 'gt.png', valid_img_visual)

    # pretrained_model_pth = 'train_log/real_train_1_log0928_%ddof_100(0)/best_model/'%num_data
    # pretrained_model_pth = 'train_log/real_id1_10000(1)_PE(arm)/best_model/'

    # ---- enrich run_info.json with training hyperparameters ----
    if RUN_INFO_PATH is not None:
        try:
            RUN_INFO["train_config"] = {
                "n_iters": int(n_iters),
                "chunksize": int(chunksize),
                "display_rate": int(display_rate),
                "center_crop": bool(center_crop),
                "center_crop_iters": int(center_crop_iters),
                "views_per_step": int(views_per_step),
                "valid_amount_eval": int(valid_amount_eval),
                "valid_views_eval": int(valid_views_eval),
                "train_amount": int(train_amount),
                "valid_amount": int(valid_amount),
                "img_height": int(height),
                "img_width": int(width),
                "Patience_threshold": int(Patience_threshold),
                "n_restarts": int(n_restarts),
            }
            _atomic_write_json(RUN_INFO_PATH, RUN_INFO)
        except Exception as e:
            print(f"[WARN] Failed to write train_config to run_info.json: {e}")

    for _ in range(n_restarts):

        model, optimizer = init_models(d_input=DOF + 3,  # DOF + 3 -> xyz and angle2 or 3 -> xyz
                                       d_filter=128,
                                       output_size=2,
                                       lr=5e-4,  # 5e-4
                                       # pretrained_model_pth=pretrained_model_pth,
                                       FLAG_PositionalEncoder = FLAG_PositionalEncoder
                                       )


        success = train(model, optimizer)
        if success:
            print('Training successful!')
            break

    # Persist final success flag (status + end_time are handled by atexit/excepthook)
    if RUN_INFO_PATH is not None:
        try:
            # `success` may be undefined if something fails early; guard with locals()
            RUN_INFO["training_success"] = bool(locals().get("success", False))
            RUN_INFO["timestamp_local"] = _now_local_iso()
            _atomic_write_json(RUN_INFO_PATH, RUN_INFO)
        except Exception as e:
            print(f"[WARN] Failed to write training_success to run_info.json: {e}")

    print(f'Done!')
    record_file_train.close()
    record_file_val.close()

'''
sanity check

python - <<'PY'
import numpy as np
p = "/data/yxk/K-data/K/fllm-sm/sim/tdcr_multiview_100.npz"
d = np.load(p)

print("keys:", d.files)
print("images:", d["images"].shape, d["images"].dtype, "min/max:", d["images"].min(), d["images"].max())
print("angles:", d["angles"].shape, d["angles"].dtype, "min/max:", d["angles"].min(), d["angles"].max())
print("rays_o:", d["rays_o"].shape, d["rays_o"].dtype)
print("rays_d:", d["rays_d"].shape, d["rays_d"].dtype)
print("near:", d["near"].shape, d["near"])
print("far :", d["far"].shape, d["far"])
PY

/data/yxk/K-data/K/fllm-sm/sim/sim_2m_no_base_new.npz

CUDA_VISIBLE_DEVICES=5 python train.py \
  --data /data/yxk/K-data/K/fllm-sm/sim/sim_2m_no_base.npz \
  --views_per_step 4 \
  --valid_amount_eval 8 \
  --valid_views_eval 0 \
  --log_dir sim_2m_no_base

CUDA_VISIBLE_DEVICES=1 python train.py \
  --data /data/yxk/K-data/K/fllm-sm/sim/sim_2m_no_base_new.npz \
  --views_per_step 4 \
  --valid_amount_eval 8 \
  --valid_views_eval 0 \
  --log_dir sim_2m_no_base_new
  

CUDA_VISIBLE_DEVICES=3 python train.py \
  --data /data/yxk/K-data/K/fllm-sm/sim/sim_2m_with_base.npz \
  --views_per_step 4 \
  --valid_amount_eval 8 \
  --valid_views_eval 0 \
  --log_dir sim_2m_with_base

CUDA_VISIBLE_DEVICES=1 python train.py \
  --data /data/yxk/K-data/K/fllm-sm/sim/sim_3m_no_base_auto_nf.npz \
  --views_per_step 4 \
  --valid_amount_eval 8 \
  --valid_views_eval 0 \
  --log_dir sim_3m_no_base

'''