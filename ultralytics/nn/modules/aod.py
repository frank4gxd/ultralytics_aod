# ultralytics/nn/modules/aod.py
# AOD blocks for Ultralytics (YOLOv8/YOLOv10) with YAML support.
# - AODImage : raw RGB (front-of-model), residual + clamp to [0,1]
# - AODFeat  : feature maps (backbone/head), residual, channel-preserving

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


# -------------------------
# helpers
# -------------------------
def _norm2d(ch: int, kind: str = "bn") -> nn.Module:
    k = (kind or "bn").lower()
    if k == "bn":
        return nn.BatchNorm2d(ch)
    if k == "gn":
        groups = max(1, min(32, ch // 4))
        return nn.GroupNorm(groups, ch)
    return nn.Identity()  # "none"


class _ConvBNAct(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int = 3,
        s: int = 1,
        p: Optional[int] = None,
        norm: str = "bn",
        act: str = "relu",
        bias: bool = False,
    ):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=bias)
        self.bn = _norm2d(c_out, norm)
        self.act = nn.ReLU(inplace=True) if (act or "relu").lower() == "relu" else nn.SiLU(inplace=True)

        # init
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# -------------------------
# cores
# -------------------------
class _AODCoreImage(nn.Module):
    """
    Lightweight image-to-image residual core.
    Uses a small 32-ch trunk; last conv is zero-initialized so residual path starts as identity.
    """
    def __init__(self, c_in: int, c_out: int, act: str = "relu", norm: str = "bn"):
        super().__init__()
        self.enc1 = _ConvBNAct(c_in, 32, 3, norm=norm, act=act)
        self.enc2 = _ConvBNAct(32, 32, 3, norm=norm, act=act)
        # tiny residual stack
        self.res = nn.Sequential(
            _ConvBNAct(32, 32, 3, norm=norm, act=act),
            _ConvBNAct(32, 32, 3, norm=norm, act=act),
        )
        self.dec = _ConvBNAct(32, 32, 3, norm=norm, act=act)
        self.out = nn.Conv2d(32, c_out, 3, padding=1, bias=True)

        # init
        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        r  = self.res(e2)
        d  = self.dec(r)
        return self.out(d)


class _AODCoreFeat(nn.Module):
    """
    Channel-preserving residual for feature maps.
    1x1 -> 3x3 -> 3x3 -> 1x1 (+norm), last 1x1 zero-init so block starts as identity.
    """
    def __init__(self, c: int, bottleneck: int = 0, norm: str = "bn", act: str = "relu"):
        super().__init__()
        h = bottleneck if bottleneck and bottleneck > 0 else max(16, min(128, c // 4))
        self.f1 = _ConvBNAct(c, h, 1, norm=norm, act=act)
        self.f2 = _ConvBNAct(h, h, 3, norm=norm, act=act)
        self.f3 = _ConvBNAct(h, h, 3, norm=norm, act=act)
        self.proj = nn.Conv2d(h, c, 1, bias=False)
        self.bn = _norm2d(c, norm)
        self.act = nn.ReLU(inplace=True) if (act or "relu").lower() == "relu" else nn.SiLU(inplace=True)

        # init
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="relu")
        with torch.no_grad():
            self.proj.weight.zero_()  # start as identity through residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.f3(self.f2(self.f1(x)))
        y = self.bn(self.proj(y))
        return self.act(x + y)


# -------------------------
# public YAML modules
# -------------------------
class AODImage(nn.Module):
    """
    AOD for raw RGB at model input. Preserves channels, optional residual + clamp.
    YAML signature: __init__(c1, c2=None, residual=True, clamp_lo=0.0, clamp_hi=1.0, detach=False, act='relu', norm='bn')
    """
    def __init__(
        self,
        c1: int,
        c2: int = None,
        residual: bool = True,
        clamp_lo: Optional[float] = 0.0,
        clamp_hi: Optional[float] = 1.0,
        detach: bool = False,
        act: str = "relu",
        norm: str = "bn",
    ):
        super().__init__()
        c2 = c1 if c2 is None else c2
        assert c1 == c2, "AODImage must preserve channels (c2 should equal c1)"
        self.core = _AODCoreImage(c_in=c1, c_out=c2, act=act, norm=norm)
        self.residual = bool(residual)
        self.detach = bool(detach)
        self.clamp = None if (clamp_lo is None or clamp_hi is None) else (float(clamp_lo), float(clamp_hi))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x.detach() if (self.detach and self.training) else x
        y = self.core(x_in)
        y = x_in + y if self.residual else y
        if self.clamp is not None:
            lo, hi = self.clamp
            y = torch.clamp(y, lo, hi)
        return y


class AODFeat(nn.Module):
    """
    AOD for feature maps (backbone/head). Preserves channels and spatial size.
    YAML signature: __init__(c1, c2=None, residual=True, norm='bn', bottleneck=0, detach=False, act='relu')
    """
    def __init__(
        self,
        c1: int,
        c2: int = None,
        residual: bool = True,
        norm: str = "bn",
        bottleneck: int = 0,
        detach: bool = False,
        act: str = "relu",
    ):
        super().__init__()
        c2 = c1 if c2 is None else c2
        assert c1 == c2, "AODFeat must preserve channels (c2 should equal c1)"
        self.core = _AODCoreFeat(c=c1, bottleneck=bottleneck, norm=norm, act=act)
        self.residual = bool(residual)
        self.detach = bool(detach)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x.detach() if (self.detach and self.training) else x
        y = self.core(x_in)
        return (x_in + y) if self.residual else y
