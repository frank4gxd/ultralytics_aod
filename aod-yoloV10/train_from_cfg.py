# train_from_cfg.py
# Run examples:
#   python train_from_cfg.py --cfg ultralytics_basics.yaml --dehaze-data dehaze_data_min.yaml
#   python train_from_cfg.py --cfg ultralytics_basics.yaml

from __future__ import annotations
import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import re
import yaml
import torch
from ultralytics import YOLO

# Local files in same folder

from wandb_log_tools import build_wandb_callbacks, register_wandb_callbacks  # interfaces only


# ===========================
# CLI
# ===========================
def parse_args():
    ap = argparse.ArgumentParser(description="Train YOLOv10n from scratch with optional AOD dehazing and W&B logging.")
    ap.add_argument("--cfg", type=str, default="./basic_config.yaml",
                    help="Main training config YAML (your ultralytics_basics.yaml)")
    ap.add_argument("--dehaze-data", type=str, default="./dataset_min.yaml",
                    help="Optional: your dehaze_data_min.yaml to convert into a standard Ultralytics data.yaml")
    ap.add_argument("--outdir", type=str, default="./runs_cfg", help="Where to put runs and generated YAMLs")
    ap.add_argument("--copy-labels", action="store_true",
                    help="Copy labels instead of symlink (useful on Windows without symlink perms)")
    return ap.parse_args()


# ===========================
# Helpers
# ===========================
def safe_symlink(src: Path, dst: Path, do_copy: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if do_copy or os.name == "nt":
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src)
        except Exception:
            shutil.copy2(src, dst)


def convert_dehaze_yaml(min_yaml: Path, outdir: Path, copy_labels: bool = False) -> Path:
    with open(min_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["data"]["data_root"]).resolve()
    tr_h = Path(cfg["data"]["train"]["hazy_images"]).resolve()
    tr_l = Path(cfg["data"]["train"]["labels"]).resolve()
    va_h = Path(cfg["data"]["val"]["hazy_images"]).resolve()
    va_l = Path(cfg["data"]["val"]["labels"]).resolve()

    def ensure_parallel_labels(images_dir: Path, labels_src_dir: Path):
        if images_dir.name != "images":
            raise ValueError(f"Expected last path element to be 'images': {images_dir}")
        labels_tgt_dir = images_dir.parent / "labels"
        labels_tgt_dir.mkdir(parents=True, exist_ok=True)
        for txt in labels_src_dir.rglob("*.txt"):
            rel = txt.relative_to(labels_src_dir)
            dst = labels_tgt_dir / rel
            safe_symlink(txt, dst, do_copy=copy_labels)
        return labels_tgt_dir

    tr_labels_tgt = ensure_parallel_labels(tr_h, tr_l)
    va_labels_tgt = ensure_parallel_labels(va_h, va_l)

    std = {
        "path": str(data_root),
        "train": str(tr_h),
        "val": str(va_h),
        "nc": int(cfg.get("nc", 1)),
        "names": cfg.get("names", ["object"]),
    }

    out_yaml = outdir / "auto_data.yaml"
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w") as f:
        yaml.safe_dump(std, f, sort_keys=False)

    print(f"[OK] Generated {out_yaml}")
    print(f"     Train images: {tr_h}")
    print(f"     Train labels: {tr_labels_tgt}")
    print(f"     Val images:   {va_h}")
    print(f"     Val labels:   {va_labels_tgt}")
    return out_yaml


def normalize_device_spec(spec) -> str:
    """
    Accepts: 'cpu', 'mps', '0', '0,1,2'
    - 'mps' (Apple GPU) requires torch.backends.mps.is_available()
    - CUDA devices require torch.cuda.is_available()
    Default if None: prefer MPS, else CUDA:0, else CPU
    """
    if spec is None:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "0" if torch.cuda.is_available() else "cpu"

    if isinstance(spec, int):
        spec = str(spec)

    s = str(spec).strip().lower()
    if s in {"cpu", "none", ""}:
        return "cpu"
    if s == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("MPS device requested but not available. Check your PyTorch build and macOS.")
        return "mps"

    # CUDA device list: '0' or '0,1,...'
    if not re.fullmatch(r"[0-9]+(,[0-9]+)*", s):
        raise ValueError(f"Invalid device spec: '{spec}'. Use 'cpu', 'mps', '0', or '0,1,...'")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but a GPU device was requested.")
    return s


# ===========================
# Main
# ===========================
def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Read main training cfg
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    data_yaml = cfg["data"]
    imgsz = int(cfg.get("imgsz", 640))
    batch = int(cfg.get("batch", 16))
    epochs = int(cfg.get("epochs", 100))
    amp = bool(cfg.get("amp", True))

    # Force YOLOv10n from scratch:
    #   Build from the YAML arch, and pass pretrained=False in train() overrides.
    model_yaml = "yolo12n_aod.yaml"

    # W&B
    wb = cfg.get("wandb", {}) or {}
    wb_enabled = bool(wb.get("enabled", False))
    wb_project = wb.get("project", "yolo")
    wb_entity = wb.get("entity", None)

    # Optional conversion of custom mapping to standard data.yaml
    if args.dehaze_data and Path(args.dehaze_data).exists():
        data_yaml = str(convert_dehaze_yaml(Path(args.dehaze_data), outdir, copy_labels=args.copy_labels))

    # Load YOLO arch (NOT weights) so we truly start from scratch
    wrapper = YOLO(model_yaml)   # build model from YAML
    model = wrapper.model


    # Device handling (CUDA / MPS / CPU)
    raw_device = cfg.get("device", None)
    device = normalize_device_spec(raw_device)
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        print("[MPS] Using Apple Metal Performance Shaders (GPU).")
    elif device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        torch.backends.cudnn.benchmark = True
        ng = len(device.split(","))
        print(f"[GPU] Using CUDA on device(s): {device}  (ngpu={ng})")
    else:
        print("[CPU] Using CPU (device='cpu').")

    # Build and register W&B callbacks (if enabled)
    if wb_enabled:
        flat_cfg = dict(
            data=data_yaml, model=model_yaml, imgsz=imgsz, batch=batch, epochs=epochs,
            device=device, amp=amp
        )
        callbacks = build_wandb_callbacks(
            run_cfg=flat_cfg,
            project=wb_project,
            entity=wb_entity,
            log_imgs_per_epoch=2,
        )
        print(f"[OK] W&B logging enabled → project='{wb_project}', entity='{wb_entity}'")
        register_wandb_callbacks(wrapper, callbacks)

    # Forward augmentation-related overrides from YAML if present
    AUG_KEYS = {
        "augment", "auto_augment", "mosaic", "close_mosaic", "mixup", "cutmix",
        "copy_paste", "copy_paste_mode", "fliplr", "flipud",
        "hsv_h", "hsv_s", "hsv_v", "translate", "scale", "shear", "perspective",
        "erasing", "rect", "multi_scale", "deterministic"
    }
    aug_overrides = {k: cfg[k] for k in AUG_KEYS if k in cfg}

    # Train (IMPORTANT: pretrained=False => from scratch)
    results = wrapper.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        amp=amp,
        cache="ram",
        workers=8,
        project=str(outdir),
        name="train",
        pretrained=False,  # FROM SCRATCH
        **aug_overrides,
    )
    print("Best weights:", getattr(wrapper, "best", None))


if __name__ == "__main__":
    main()
