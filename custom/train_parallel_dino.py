# train_parallel_dino_fixed.py — YOLOv12 + parallel DINO (HARD FREEZE + STABLE ARGS)
from __future__ import annotations
import argparse, os, sys, random
from typing import Any, Dict, List

import numpy as np
import torch
from ultralytics import YOLO

# Drop-in backbone module (this file lives next to this script as parallel_dino.py)
from parallel_dino import DINOEncoder, Take, DINOYolo12Backbone

# Optional helpers
try:
    from wandb_log_dino import build_wandb_callbacks, register_wandb_callbacks
except Exception:
    build_wandb_callbacks = register_wandb_callbacks = None

try:
    from local_preview_saver import build_local_preview_callbacks
except Exception:
    build_local_preview_callbacks = None


# ----------------------------
# ENFORCED FREEZE
# ----------------------------
def enforce_dino_freeze(model, freeze: bool = True) -> bool:
    """
    HARD FREEZE: set requires_grad=False for all 'dino.encoder' params
    and mark them with a persistent attribute so we can later strip them from the optimizer.
    """
    frozen_count = 0
    total_count = 0
    for name, p in model.named_parameters():
        if "dino.encoder" in name:
            total_count += 1
            p.requires_grad = not freeze
            # sticky flag we rely on for optimizer cleanup
            setattr(p, "_is_frozen", bool(freeze))
            if freeze:
                frozen_count += 1

    # eval()/train() on the encoder
    for m in model.modules():
        if hasattr(m, "dino") and hasattr(m.dino, "encoder"):
            if freeze:
                m.dino.encoder.eval()
            else:
                m.dino.encoder.train()

    print(f"[ENFORCED FREEZE] {frozen_count}/{total_count} DINO parameters frozen")
    return (frozen_count == total_count) if freeze else True


def check_dino_frozen(model) -> bool:
    frozen_params = 0
    total_params = 0
    trainable_params: List[str] = []
    for name, p in model.named_parameters():
        if "dino.encoder" in name:
            total_params += 1
            if not p.requires_grad:
                frozen_params += 1
            else:
                trainable_params.append(name)
    ok = (frozen_params == total_params)
    print(f"[FREEZE CHECK] {frozen_params}/{total_params} DINO params frozen - {'✓ OK' if ok else '✗ FAILED'}")
    if trainable_params:
        print(f"[WARNING] Trainable DINO params: {trainable_params[:3]}{'...' if len(trainable_params) > 3 else ''}")
    return ok


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser("Train YOLOv12 with parallel DINO branch (HARD FREEZE)")
    ap.add_argument("--cfg", type=str, default="train_parallel_dino.yaml")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--weights", type=str, default=None)

    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--project", type=str, default=None)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--workers", type=int, default=None)

    ap.add_argument("--amp", type=str, default=None, help="true/false; default False in this script")

    ap.add_argument("--debug-ch", action="store_true", help="print in/out channels after model build")
    ap.add_argument("--no-previews", action="store_true", help="disable local preview callbacks")

    # Fusion & DINO options
    ap.add_argument("--fuse", type=str, default="concat",
                    choices=["concat", "sum", "adaptive", "spatial", "dynamic", "multiscale"],
                    help="fusion mode (sum = fastest)")
    ap.add_argument("--dino-imn", dest="dino_imn", type=str, default="true",
                    help="use ImageNet normalization for DINO")
    ap.add_argument("--freeze-dino", dest="freeze_dino", type=str, default="true",
                    help="freeze DINO encoder weights")
    ap.add_argument("--dino-name", type=str, default="vit_small_patch14_dinov2",
                    help="DINO model name for timm (lighter default)")

    # Seeding & determinism
    ap.add_argument("--seed", type=int, default=0, help="global random seed")
    ap.add_argument("--deterministic", type=str, default="true", help="deterministic mode")

    # Performance tuning
    ap.add_argument("--optimize-speed", action="store_true", help="apply speed optimizations")
    return ap.parse_args()


# ----------------------------
# Utils
# ----------------------------
def _coerce_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(x)


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _register_custom_layers():
    import ultralytics.nn.tasks as yolo_tasks
    import ultralytics.nn.modules as yolo_modules
    for mod in (yolo_tasks, yolo_modules):
        setattr(mod, "DINOYolo12Backbone", DINOYolo12Backbone)
        setattr(mod, "DINOEncoder", DINOEncoder)
        setattr(mod, "Take", Take)
    print("[OK] Registered custom layers: DINOYolo12Backbone, DINOEncoder, Take")


def _find_first_module(root_module, cls):
    for m in root_module.modules():
        if isinstance(m, cls):
            return m
    return None

def _debug_model_channels(model):
    print("\n=== Debugging Model Channels ===")
    for name, module in model.named_modules():
        if hasattr(module, "out_channels"):
            print(f"{name}: out_channels = {module.out_channels}")
        if hasattr(module, "in_channels"):
            print(f"{name}: in_channels = {module.in_channels}")
    print("=== End Debug ===\n")


def _set_global_seed(seed: int, deterministic: bool = True):
    print(f"[OK] Seeding: seed={seed}, deterministic={deterministic}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def _swap_backbone(model: YOLO, *, fuse_mode: str = "sum", dino_imagenet_norm: bool = True,
                   dino_pretrained: bool = True, dino_freeze: bool = True,
                   dino_name: str = "vit_small_patch14_dinov2"):
    """Replace YOLO backbone with the DINO-parallel backbone"""
    new_backbone = DINOYolo12Backbone(
        dino_name=dino_name,
        dino_pretrained=dino_pretrained,
        dino_freeze=dino_freeze,       # enforced again below
        dino_imagenet_norm=dino_imagenet_norm,
        fuse_mode=fuse_mode,
    )
    model.model.model[0] = new_backbone
    print(f"[OK] Swapped backbone → fuse_mode={fuse_mode}, IMN={dino_imagenet_norm}, "
          f"freeze={dino_freeze}, name={dino_name}")
    return new_backbone


def _check_ram() -> bool:
    try:
        import psutil
        return psutil.virtual_memory().available > 8 * 1024 ** 3  # >8GB avail
    except Exception:
        return False


def _stable_train_args(args, cfg) -> Dict[str, Any]:
    # conservative defaults for stability
    amp_flag = _coerce_bool(args.amp) if args.amp is not None else False
    return {
        'data': args.data or cfg.get("data", "dataset_dino.yaml"),
        'imgsz': args.imgsz or cfg.get("imgsz", 512),
        'batch': args.batch or cfg.get("batch", 16),
        'epochs': args.epochs or cfg.get("epochs", 100),
        'device': args.device or cfg.get("device", "0"),
        'workers': args.workers or cfg.get("workers", 0),     # 0 for determinism / Windows stability

        # Stability
        'amp': amp_flag,                 # default False here
        'rect': True,
        'nms': True,
        'cache': 'disk',                 # avoid RAM-cache non-determinism
        'deterministic': True,
        'seed': args.seed,

        # Optim
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'warmup_epochs': 5,

        # Light augs
        'auto_augment': 'none',
        'erasing': 0.0,
        'mosaic': 0.3,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'close_mosaic': 5,
        'hsv_h': 0.01,
        'hsv_s': 0.2,
        'hsv_v': 0.2,
        'translate': 0.05,

        # Output
        'project': args.project or cfg.get("project", "./runs_fast"),
        'name': args.name or cfg.get("name", "train_dino_fast"),
        'save': True,
        'pretrained': False,
    }


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    cfg = _load_yaml(args.cfg)

    # set device env early
    device = str(args.device or cfg.get("device", "0"))
    if device and device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = device

    # seed
    seed = int(args.seed)
    deterministic = _coerce_bool(args.deterministic)
    _set_global_seed(seed, deterministic)

    # files
    model_yaml = args.model or cfg.get("model", "yolo12_parallel_dino.yaml")
    data_yaml = args.data or cfg.get("data", "dataset_dino.yaml")
    weights = args.weights or cfg.get("weights", None)

    # register + build
    _register_custom_layers()
    yolo = YOLO(model_yaml)

    # swap backbone
    _ = _swap_backbone(
        yolo,
        fuse_mode=args.fuse,
        dino_imagenet_norm=_coerce_bool(args.dino_imn),
        dino_pretrained=True,
        dino_freeze=_coerce_bool(args.freeze_dino),
        dino_name=args.dino_name,
    )

    # ENFORCED FREEZE (now)
    freeze_flag = _coerce_bool(args.freeze_dino)
    enforce_dino_freeze(yolo.model, freeze_flag)
    check_dino_frozen(yolo.model)

    def _on_pretrain_start(trainer):
        # Model may not be built yet (can be a str). Just skip.
        m = getattr(trainer, "model", None)
        if not hasattr(m, "named_parameters"):
            return
        enforce_dino_freeze(m, freeze_flag)
        check_dino_frozen(m)

    def _on_pretrain_end(trainer):
        # Now the model and optimizer should exist — enforce + scrub optimizer
        m = getattr(trainer, "model", None)
        opt = getattr(trainer, "optimizer", None)
        if hasattr(m, "named_parameters"):
            enforce_dino_freeze(m, freeze_flag)
            check_dino_frozen(m)
        if opt is not None:
            # drop any frozen params from param_groups so they never get updated
            for g in opt.param_groups:
                g["params"] = [p for p in g["params"] if not getattr(p, "_is_frozen", False)]

    def _on_fit_start(trainer):
        # Extra belt-and-suspenders
        m = getattr(trainer, "model", None)
        if hasattr(m, "named_parameters"):
            enforce_dino_freeze(m, freeze_flag)
            check_dino_frozen(m)

    def _on_resume(trainer):
        m = getattr(trainer, "model", None)
        if hasattr(m, "named_parameters"):
            enforce_dino_freeze(m, freeze_flag)
            check_dino_frozen(m)

    yolo.add_callback('on_pretrain_routine_start', _on_pretrain_start)
    yolo.add_callback('on_pretrain_routine_end', _on_pretrain_end)
    yolo.add_callback('on_fit_start', _on_fit_start)
    yolo.add_callback('on_resume', _on_resume)
    print("[OK] DINO freeze enforcement registered")

    if args.debug_ch:
        _debug_model_channels(yolo.model)

    # local preview callbacks (optional)
    if not args.no_previews and build_local_preview_callbacks is not None:
        dino_mod_for_vis = _find_first_module(yolo.model, DINOEncoder)
        if dino_mod_for_vis is not None:
            local_cbs = build_local_preview_callbacks(
                dino_module=dino_mod_for_vis,
                num_samples=2,
                every_n_epochs=1,
                save_hazy=True,
                save_dino_heatmap=True,
                save_dino_tiles=False,
                dino_channels=4,
                dino_reduce="mean",
                subdir="previews",
            )
            for name_cb, fn in local_cbs.items():
                yolo.add_callback(name_cb, fn)
            print("[OK] Local preview callbacks registered.")
        else:
            print("[INFO] DINOEncoder not found; previews disabled.")
    else:
        print("[INFO] Previews disabled by flag or helper missing.")

    # warm-start
    if weights:
        try:
            yolo.load(weights)
            enforce_dino_freeze(yolo.model, freeze_flag)
            print(f"[OK] Loaded warm-start weights: {weights}")
        except Exception as e:
            print(f"[WARN] Could not load weights '{weights}': {e}")

    # W&B (optional)
    wb_cfg = cfg.get("wandb", {}) or {}
    if wb_cfg.get("enabled", False) and build_wandb_callbacks is not None and register_wandb_callbacks is not None:
        callbacks = build_wandb_callbacks(
            run_cfg=dict(
                cfg_path=str(args.cfg),
                model=model_yaml,
                data=data_yaml,
                weights=weights,
                imgsz=args.imgsz or cfg.get("imgsz", 512),
                batch=args.batch or cfg.get("batch", 16),
                epochs=args.epochs or cfg.get("epochs", 100),
                device=device,
                workers=args.workers or cfg.get("workers", 0),
                amp=_coerce_bool(args.amp) if args.amp is not None else False,
                project=args.project or cfg.get("project", "./runs_fast"),
                name=args.name or cfg.get("name", "train_dino_fast"),
                seed=seed,
            ),
            project=wb_cfg.get("project", "yolo"),
            entity=wb_cfg.get("entity", None),
            dehaze_module=None,
            log_imgs_per_epoch=int(wb_cfg.get("log_imgs_per_epoch", 2)),
            extra={
                "model_yaml": model_yaml,
                "data_yaml": data_yaml,
                "notes": cfg.get("notes", ""),
                "extra_files": wb_cfg.get("extra_files", []) or [],
            },
        )
        register_wandb_callbacks(yolo, callbacks)
        print(f"[OK] W&B logging enabled")

    # Train
    train_args = _stable_train_args(args, cfg)
    print("\n=== TRAIN CONFIG (stable) ===")
    for k, v in train_args.items():
        if k not in ["data"]:
            print(f"  {k}: {v}")
    print("=================================\n")
    yolo.train(**train_args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"[FATAL] {e}")
        raise
