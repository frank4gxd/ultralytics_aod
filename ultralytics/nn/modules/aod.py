# ultralytics/nn/modules/aod.py
import torch
import torch.nn as nn
from typing import Optional


class ResidualConvBlock(nn.Module):
    def __init__(self, ch: int, k: int = 3, p: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, k, padding=p)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch, ch, k, padding=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.conv1(x))
        y = self.conv2(y)
        return self.relu(y + x)


class AODImage(nn.Module):
    """
    Image-space AOD block for RGB input.
    IMPORTANT: signature matches Ultralytics parser: (c1, c2, *args).
    We **force** channel preservation by setting c2=c1 internally.
    """
    def __init__(
        self,
        c1: int,
        c2: Optional[int] = None,
        residual: bool = True,
        clamp_lo: float = 0.0,
        clamp_hi: float = 1.0,
        detach: bool = False,
    ):
        super().__init__()
        self.c1 = int(c1)
        self.c2 = int(c1)  # force preserve channels regardless of c2 passed in
        self.residual = residual
        self.clamp = (float(clamp_lo), float(clamp_hi))
        self.detach = detach

        mid = 32
        self.enc1 = nn.Conv2d(self.c1, mid, 3, padding=1)
        self.enc2 = nn.Conv2d(mid, mid, 3, padding=1)
        self.block = ResidualConvBlock(mid)
        self.dec1 = nn.Conv2d(mid, mid, 3, padding=1)
        self.out = nn.Conv2d(mid, self.c1, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x.detach() if (self.detach and self.training) else x
        e1 = self.act(self.enc1(x_in))
        e2 = self.act(self.enc2(e1))
        b = self.block(e2)
        d1 = self.act(self.dec1(b))
        y = self.out(d1)
        y = x_in + y if self.residual else y
        lo, hi = self.clamp
        return torch.clamp(y, lo, hi)


class AODFeat(nn.Module):
    """
    Feature-space AOD block, channel-preserving.
    Accepts either:
      - AODFeat(c1, c2, residual, norm, bottleneck, detach)   # explicit c2 (Ultralytics-style)
      - AODFeat(c1, residual, norm, bottleneck, detach)       # no c2 (short form)
    We'll parse *args by type to be robust against parser differences.
    """
    def __init__(self, c1: int, *args):
        super().__init__()
        self.c1 = int(c1)

        # Defaults
        c2: Optional[int] = None
        residual: bool = True
        norm: str = "bn"
        bottleneck: int = 0
        detach: bool = False

        # ---- Type-driven parsing (order-independent within reason)
        a = list(args)

        # optional leading c2 (int)
        if a and isinstance(a[0], int):
            c2 = int(a.pop(0))  # we still preserve channels internally

        # residual (bool)
        if a and isinstance(a[0], bool):
            residual = bool(a.pop(0))

        # norm (str)
        if a and isinstance(a[0], str):
            norm = str(a.pop(0)).lower()

        # bottleneck (int)
        if a and isinstance(a[0], int):
            bottleneck = int(a.pop(0))

        # detach (bool)
        if a and isinstance(a[0], bool):
            detach = bool(a.pop(0))

        # Fallback safety if anything odd slipped through
        if not isinstance(norm, str):
            norm = "bn"
        norm = norm.lower()

        self.c2 = self.c1               # preserve channels regardless of c2 passed
        self.residual = residual
        self.detach = detach

        # bottleneck width (auto if 0)
        h = bottleneck if bottleneck > 0 else max(16, min(128, self.c1 // 4))

        # Norm layers (GN for small batch, BN otherwise)
        if norm == "gn":
            groups_h = min(32, max(1, h // 8))
            groups_c = min(32, max(1, self.c1 // 8))
            n1 = nn.GroupNorm(groups_h, h)
            n2 = nn.GroupNorm(groups_c, self.c1)
        else:
            n1 = nn.BatchNorm2d(h)
            n2 = nn.BatchNorm2d(self.c1)

        self.proj1 = nn.Conv2d(self.c1, h, 1, bias=False)
        self.dw    = nn.Conv2d(h, h, 3, padding=1, groups=h, bias=False)  # depthwise smoothing
        self.act   = nn.ReLU(inplace=True)
        self.proj2 = nn.Conv2d(h, self.c1, 1, bias=False)
        self.n1, self.n2 = n1, n2

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x.detach() if (self.detach and self.training) else x
        y = self.proj1(x_in)
        y = self.n1(y)
        y = self.act(y)
        y = self.dw(y)
        y = self.act(y)
        y = self.proj2(y)
        y = self.n2(y)
        return x_in + y if self.residual else y
