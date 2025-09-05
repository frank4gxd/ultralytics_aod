
# Ultra-detailed W&B logging for Ultralytics v8/v10.
# Logs batch/epoch scalars, LR, grad norms, hazy->dehazed previews, and saved plots.

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import numpy as np

try:
    import wandb
    _HAVE_WANDB = True
except Exception:
    _HAVE_WANDB = False

try:
    from torchvision.utils import make_grid
except Exception:
    make_grid = None

def _as_float(v):
    """Return float for Python/NumPy/Torch scalars; None if not scalar."""
    try:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.ndarray) and v.shape == ():
            return float(v.item())
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                return float(v.item())
            # fall back to mean for tiny tensors (shouldn't happen here)
            return float(v.detach().mean().item())
    except Exception:
        pass
    return None

def _to_numpy_uint8(img_t: torch.Tensor):
    def one(x):
        x = x.detach().float().clamp(0, 1).cpu().numpy()
        if x.ndim == 3:
            x = np.transpose(x, (1, 2, 0))
        return (x * 255.0 + 0.5).astype("uint8")
    if img_t.ndim == 4:
        return [one(t) for t in img_t]
    return one(img_t)


def _remap_val_metric_keys(d: dict) -> dict:
    if not isinstance(d, dict):
        return {}
    out = {}
    alias = {
        # common YOLOv8/10 keys
        "metrics/mAP50(B)": "mAP50",
        "metrics/mAP50": "mAP50",
        "metrics/mAP50-95(B)": "mAP50_95",
        "metrics/mAP50-95": "mAP50_95",
        "metrics/precision(B)": "precision",
        "metrics/precision": "precision",
        "metrics/recall(B)": "recall",
        "metrics/recall": "recall",
        # older-style keys (just in case)
        "metrics/mAP_0.5(B)": "mAP50",
        "metrics/mAP_0.5": "mAP50",
        "metrics/mAP_0.5:0.95(B)": "mAP50_95",
        "metrics/mAP_0.5:0.95": "mAP50_95",
    }
    for k, v in d.items():
        name = alias.get(k)
        if name is None:
            continue
        fv = _as_float(v)
        if fv is not None:
            out[name] = fv
    return out


def _lr_from_trainer(trainer) -> float:
    try:
        return float(trainer.optimizer.param_groups[0]["lr"])
    except Exception:
        return float("nan")


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.data
            total += float(g.norm(2).item() ** 2)
    return float(total ** 0.5)


def _gather_scalar_metrics(trainer) -> Dict[str, float]:
    out: Dict[str, float] = {}

    # common trainer attributes
    for k in ("box_loss", "cls_loss", "dfl_loss", "seg_loss", "pose_loss", "loss", "lr"):
        v = getattr(trainer, k, None)
        fv = _as_float(v)
        if fv is not None:
            out[k] = fv

    # trainer.metrics (dict)
    mdict = getattr(trainer, "metrics", None)
    if isinstance(mdict, dict):
        for k, v in mdict.items():
            fv = _as_float(v)
            if fv is not None:
                out.setdefault(k, fv)

    # validator metrics
    vld = getattr(getattr(trainer, "validator", None), "metrics", None)
    if hasattr(vld, "results_dict"):
        for k, v in vld.results_dict.items():
            fv = _as_float(v)
            if fv is not None:
                out.setdefault(k, fv)

    # per-epoch loss tuple/list
    loss_items = getattr(trainer, "loss_items", None)
    if isinstance(loss_items, (list, tuple)):
        for i, v in enumerate(loss_items):
            fv = _as_float(v)
            if fv is not None:
                out.setdefault(f"loss_items/{i}", fv)

    return out

def _find_plot_images(save_dir: Path) -> List[Path]:
    if not save_dir or not save_dir.exists():
        return []
    pats = ["train_batch*.jpg", "val_batch*.jpg", "confusion_matrix.png",
            "PR_curve.png", "F1_curve.png", "P_curve.png", "R_curve.png",
            "results.png"]
    files: List[Path] = []
    for pat in pats:
        files.extend(save_dir.glob(pat))
    files.sort(key=lambda p: p.stat().st_mtime)
    return files[-12:]


def register_wandb_callbacks(wrapper, cbs: dict | None):
    """Register callbacks safely; don't pass callbacks= to train()."""
    if not cbs:
        return
    for name, fn in cbs.items():
        try:
            wrapper.add_callback(name, fn)
        except Exception as e:
            print(f"[WARN] add_callback failed for {name}: {e}")
    # normalize to lists
    try:
        cb = getattr(wrapper, "callbacks", None)
        if isinstance(cb, dict):
            for k, v in list(cb.items()):
                if callable(v):
                    cb[k] = [v]
                elif isinstance(v, (list, tuple)):
                    cb[k] = [f for f in v if callable(f)]
                else:
                    cb[k] = []
    except Exception as e:
        print(f"[WARN] normalize callbacks failed: {e}")


def build_wandb_callbacks(
    run_cfg: Dict[str, Any],
    project: str,
    entity: Optional[str],
    dehaze_module: Optional[torch.nn.Module] = None,
    log_imgs_per_epoch: int = 2,
):
    if not _HAVE_WANDB:
        print("[W&B] wandb not installed; logging disabled.")
        return {}

    state = {"run": None, "step": 0}

    def _ensure_run(trainer):
        # reuse existing if already inited elsewhere
        if state["run"] is not None or getattr(wandb, "run", None) is not None:
            state["run"] = getattr(wandb, "run", state["run"])
            return
        name = f"{Path(getattr(trainer, 'save_dir', 'run')).name}"
        run = wandb.init(project=project, entity=entity, config=run_cfg, name=name, reinit=True)
        state["run"] = run
        try:
            wandb.watch(trainer.model, log="all", log_freq=500)
        except Exception:
            pass
        try:
            nparams = sum(p.numel() for p in trainer.model.parameters())
            wandb.log({"model/params": nparams, "epoch": 0}, step=0)
        except Exception:
            pass

    def _log_images(trainer, batch: Optional[dict]):
        if log_imgs_per_epoch <= 0:
            return
        if batch is None or "img" not in batch:
            return
        imgs = batch["img"]
        if not isinstance(imgs, torch.Tensor):
            return
        b = min(log_imgs_per_epoch, imgs.shape[0])
        hazy = imgs[:b]
        try:
            dehazed = None
            if dehaze_module is not None:
                dehaze_module.eval()
                with torch.no_grad():
                    dehazed = dehaze_module(hazy.to(next(dehaze_module.parameters()).device)).cpu()
        except Exception:
            dehazed = None

        hz_np = _to_numpy_uint8(hazy)
        hz_list = hz_np if isinstance(hz_np, list) else [hz_np]
        dz_list = None
        if dehazed is not None:
            dz_np = _to_numpy_uint8(dehazed)
            dz_list = dz_np if isinstance(dz_np, list) else [dz_np]

        panels = []
        for i in range(len(hz_list)):
            if dz_list is not None and make_grid is not None:
                import numpy as np
                pair = torch.stack([
                    torch.from_numpy(np.transpose(hz_list[i], (2, 0, 1))).float() / 255.0,
                    torch.from_numpy(np.transpose(dz_list[i], (2, 0, 1))).float() / 255.0,
                ])
                grid = make_grid(pair, nrow=2)
                panels.append(wandb.Image(_to_numpy_uint8(grid), caption="hazy → dehazed"))
            else:
                panels.append(wandb.Image(hz_list[i], caption="hazy"))
        if panels:
            wandb.log({"preview/hazy_dehazed": panels}, step=state["step"])

    # ---- Callbacks ----
    def on_fit_start(trainer):
        _ensure_run(trainer)

    def on_train_start(trainer):
        _ensure_run(trainer)

    def on_train_epoch_start(trainer):
        _ensure_run(trainer)
        state["step"] = int(getattr(trainer, "epoch", 0))
        wandb.log({"lr": _lr_from_trainer(trainer), "epoch": getattr(trainer, "epoch", 0)}, step=state["step"])

    def on_train_batch_end(trainer):
        _ensure_run(trainer)
        epoch = int(getattr(trainer, "epoch", 0))
        batch_i = int(getattr(trainer, "batch_i", 0))
        n_batches = int(getattr(trainer, "nb", batch_i + 1))
        state["step"] = epoch * max(1, n_batches) + batch_i
        scalars = _gather_scalar_metrics(trainer)
        scalars.update({"lr": _lr_from_trainer(trainer), "epoch": epoch})
        wandb.log({f"train/{k}": v for k, v in scalars.items()}, step=state["step"])
        if batch_i == 0:
            _log_images(trainer, getattr(trainer, "batch", None))

    def on_train_epoch_end(trainer):
        _ensure_run(trainer)
        try:
            gnorm = _grad_norm(trainer.model)
            wandb.log({"train/grad_norm": gnorm, "epoch": getattr(trainer, "epoch", 0)}, step=state["step"])
        except Exception:
            pass

    def on_val_end(trainer):
        _ensure_run(trainer)
        scalars = _gather_scalar_metrics(trainer)
        epoch = int(getattr(trainer, "epoch", 0))
        scalars.update({"epoch": epoch})
        wandb.log({f"val/{k}": v for k, v in scalars.items()}, step=state["step"])
        # nicer names
        try:
            vld = getattr(getattr(trainer, "validator", None), "metrics", None)
            if hasattr(vld, "results_dict"):
                nice = _remap_val_metric_keys(vld.results_dict)
                if nice:
                    wandb.log({f"val/{k}": v for k, v in nice.items()}, step=state["step"])
                    if "mAP50" in nice and "mAP50_95" in nice:
                        print(f"[VAL][epoch {epoch}] mAP50={nice['mAP50']:.4f}, mAP50-95={nice['mAP50_95']:.4f}, "
                              f"precision={nice.get('precision', float('nan')):.4f}, "
                              f"recall={nice.get('recall', float('nan')):.4f}")
        except Exception:
            pass
        # plots
        try:
            save_dir = Path(getattr(trainer, "save_dir", ""))
            imgs = _find_plot_images(save_dir)
            if imgs:
                wandb.log({"val/plots": [wandb.Image(str(p)) for p in imgs]}, step=state["step"])
        except Exception:
            pass

    def on_fit_end(trainer):
        _ensure_run(trainer)
        try:
            best = getattr(trainer, "best_fitness", None)
            if best is not None:
                wandb.summary["best_fitness"] = float(best)
        except Exception:
            pass
        run = state.get("run", None)
        if run is not None:
            run.finish()

    return {
        "on_train_start": on_train_start,
        "on_fit_start": on_fit_start,
        "on_train_epoch_start": on_train_epoch_start,
        "on_train_batch_end": on_train_batch_end,
        "on_train_epoch_end": on_train_epoch_end,
        "on_val_end": on_val_end,
        "on_fit_end": on_fit_end,
    }
