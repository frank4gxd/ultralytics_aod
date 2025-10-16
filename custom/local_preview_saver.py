# local_preview_saver.py
# Save random few previews per epoch to the current YOLO run directory (no W&B).

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List
import random
import math

import torch
import numpy as np

try:
    from torchvision.utils import make_grid
    _HAVE_TV = True
except Exception:
    _HAVE_TV = False

try:
    import imageio.v2 as imageio
    _HAVE_IMGIO = True
except Exception:
    _HAVE_IMGIO = False


def _to_numpy_uint8(img_t: torch.Tensor):
    """Tensor [C,H,W] or [H,W] in [0,1] -> HxW(xC) uint8."""
    x = img_t.detach().float().cpu().numpy()
    if x.ndim == 3:
        x = np.transpose(x, (1, 2, 0))
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype("uint8")


def _save_png(path: Path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    if _HAVE_IMGIO:
        imageio.imwrite(str(path), arr)
    else:
        # Very minimal fallback using PIL if available
        try:
            from PIL import Image
            Image.fromarray(arr).save(str(path))
        except Exception as e:
            print(f"[WARN] Could not save {path}: {e}")


def _reduce_to_heatmap(feat: torch.Tensor, how: str = "mean") -> torch.Tensor:
    """feat: [N,C,H,W] -> [1,1,H,W] normalized."""
    f = feat[:1]
    if how == "max":
        hm = f.max(dim=1, keepdim=True).values
    else:
        hm = f.mean(dim=1, keepdim=True)
    hm = hm - hm.min()
    hm = hm / (hm.max() + 1e-6)
    return hm


def _tile_channels(feat: torch.Tensor, max_channels: int = 4) -> torch.Tensor:
    """
    Take first K channels from feat [N,C,H,W] -> [C,H,W] normalized (first image).
    """
    f = feat[:1]
    c = min(max_channels, f.shape[1])
    c = max(1, c)
    f = f[:, :c, :, :].clone()  # [1,c,H,W]
    eps = 1e-6
    for i in range(c):
        ch = f[0, i]
        ch = ch - ch.min()
        ch = ch / (ch.max() + eps)
        f[0, i] = ch
    return f[0]  # [c,H,W]


def build_local_preview_callbacks(
    dino_module: Optional[torch.nn.Module] = None,
    num_samples: int = 2,              # save up to N random images per epoch from the first train batch
    every_n_epochs: int = 1,           # save every k epochs
    save_hazy: bool = True,            # save input images
    save_dino_heatmap: bool = True,    # save DINO heatmaps
    save_dino_tiles: bool = True,      # save DINO channel tiles
    dino_channels: int = 4,            # number of channels to tile if tiles enabled
    dino_reduce: str = "mean",         # "mean" | "max"
    subdir: str = "previews",          # subfolder under save_dir
) -> dict:
    """
    Returns a dict of Ultralytics callbacks that save local previews during training.
    """

    state = {
        "saved_epoch": -1,  # to ensure we only save once per epoch (on the first batch)
    }

    def _epoch_should_save(epoch: int) -> bool:
        if every_n_epochs <= 0:
            return False
        return (epoch % every_n_epochs) == 0

    def _save_one_image(img: torch.Tensor, out_dir: Path, tag: str, epoch: int, idx: int):
        # img: [3,H,W] in [0,1]
        out = _to_numpy_uint8(img)
        _save_png(out_dir / f"epoch_{epoch:03d}_{tag}_{idx:02d}.png", out)

    def _save_dino_feats(x: torch.Tensor, out_dir: Path, epoch: int, index_tag: str):
        if dino_module is None:
            return
        try:
            dino_module.eval()
            with torch.no_grad():
                dev = next(dino_module.parameters()).device
                c3, c4, c5 = dino_module(x.unsqueeze(0).to(dev))  # run one image
        except Exception as e:
            print(f"[WARN] DINO forward failed: {e}")
            return

        # Heatmaps
        if save_dino_heatmap:
            for tag, f in (("C3", c3), ("C4", c4), ("C5", c5)):
                hm = _reduce_to_heatmap(f, how=dino_reduce)  # [1,1,H,W]
                arr = _to_numpy_uint8(hm[0])  # [H,W]
                _save_png(out_dir / f"epoch_{epoch:03d}_DINO_{tag}_{dino_reduce}_{index_tag}.png", arr)

        # Channel tiles
        if save_dino_tiles and _HAVE_TV:
            import torch as _torch
            tiles = []
            for tag, f in (("C3", c3), ("C4", c4), ("C5", c5)):
                t = _tile_channels(f, dino_channels)  # [C,H,W]
                # build grid (near-square)
                c = t.shape[0]
                nrow = max(1, int(math.ceil(math.sqrt(c))))
                # make_grid expects BCHW; create [C,1,H,W] and let make_grid tile channels like images
                grid = make_grid(t.unsqueeze(1), nrow=nrow)  # [3?,H,W] but our "images" are single-channel
                # If grid has 1 channel, expand to 3 for PNG
                if grid.shape[0] == 1:
                    grid = _torch.cat([grid, grid, grid], dim=0)
                arr = _to_numpy_uint8(grid)
                _save_png(out_dir / f"epoch_{epoch:03d}_DINO_{tag}_ch{dino_channels}_{index_tag}.png", arr)

    # --- Ultralytics callback: run once per epoch on first train batch ---
    def on_train_batch_end(trainer):
        epoch = int(getattr(trainer, "epoch", 0))
        batch_i = int(getattr(trainer, "batch_i", 0))

        # Only on first batch of the epoch, and only at the desired frequency
        if batch_i != 0 or state["saved_epoch"] == epoch or not _epoch_should_save(epoch):
            return

        batch = getattr(trainer, "batch", None)
        if not batch or "img" not in batch:
            return

        imgs: torch.Tensor = batch["img"]
        if not isinstance(imgs, torch.Tensor) or imgs.ndim != 4:
            return

        save_dir = Path(getattr(trainer, "save_dir", ".")) / subdir
        save_dir.mkdir(parents=True, exist_ok=True)

        # Randomly sample up to num_samples indices from this batch
        bsz = imgs.shape[0]
        k = min(max(1, int(num_samples)), bsz)
        picks = random.sample(range(bsz), k=k)

        # Save hazy images
        if save_hazy:
            for idx in picks:
                _save_one_image(imgs[idx], save_dir, "hazy", epoch, idx)

        # Save DINO viz for the FIRST sampled index (popular choice: 1 per epoch)
        if dino_module is not None:
            idx0 = picks[0]
            try:
                _save_dino_feats(imgs[idx0], save_dir, epoch, index_tag=f"idx{idx0:02d}")
            except Exception as e:
                print(f"[WARN] DINO visualization save failed: {e}")

        state["saved_epoch"] = epoch

    # You can add more callbacks if you want (e.g., on_val_end), but this one is enough for previews.
    return {
        "on_train_batch_end": on_train_batch_end
    }
