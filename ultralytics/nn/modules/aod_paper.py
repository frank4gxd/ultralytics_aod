# ultralytics/nn/modules/aod_paper.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------- PONO & MS ----------
class PONO(nn.Module):
    """Positional Normalization: normalize across channels at each (H,W) position."""
    def __init__(self, affine: bool = False, eps: float = 1e-5):
        super().__init__()
        self.affine = affine
        self.eps = eps
        if affine:
            # 延后初始化：运行时按输入尺寸注册参数（一般不需要）
            self.gamma = None
            self.beta = None

    def forward(self, x: torch.Tensor):
        # mean/std across C (keep dims for broadcasting)
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt()
        y = (x - mean) / std
        if self.affine and (self.gamma is not None) and (self.beta is not None):
            y = y * self.gamma + self.beta
        return y, mean, std


class MS(nn.Module):
    """Moment Shortcut: re-inject (mean, std) later as y*std + mean."""
    def forward(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        return x * std + mean


# ---------- K-estimation: 5 convs + multi-scale concat ----------
class KEstimationAOD(nn.Module):
    """
    Paper/GitHub-style AOD K-estimation:
      conv1: 1x1, in C -> 3 (ReLU)
      conv2: 3x3, in C -> 3 (ReLU)  [可选：与部分复现一致的“conv2从x1输入”开关]
      cat1 = [c1, c2] -> 6
      conv3: 5x5, in 6 -> 3 (ReLU)
      cat2 = [c2, c3] -> 6
      conv4: 7x7, in 6 -> 3 (ReLU)
      cat3 = [c1, c2, c3, c4] -> 12
      conv5: 3x3, in 12 -> C (LINEAR; no ReLU)
    """
    def __init__(self, in_ch: int = 3, conv2_from_x1: bool = False):
        super().__init__()
        C = in_ch
        self.conv1 = nn.Conv2d(C, 3, 1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(C, 3, 3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(6, 3, 5, padding=2, bias=True)
        self.conv4 = nn.Conv2d(6, 3, 7, padding=3, bias=True)
        self.conv5 = nn.Conv2d(12, C, 3, padding=1, bias=True)
        self.conv2_from_x1 = bool(conv2_from_x1)

    def forward(self, x: torch.Tensor):
        c1 = F.relu(self.conv1(x))
        # 有的 GitHub 复现把 conv2 输入写成 x1；默认更贴近论文/多数端口：conv2(x)
        c2_in = c1 if self.conv2_from_x1 else x
        c2 = F.relu(self.conv2(c2_in))

        cat1 = torch.cat((c1, c2), 1)
        c3 = F.relu(self.conv3(cat1))

        cat2 = torch.cat((c2, c3), 1)
        c4 = F.relu(self.conv4(cat2))

        cat3 = torch.cat((c1, c2, c3, c4), 1)
        K = self.conv5(cat3)  # linear
        return K


# ---------- Core AOD (Paper) ----------
class AODNetPaperCore(nn.Module):
    """J = K * I - K + b, then clamp to [lo, hi] if set."""
    def __init__(self, in_ch: int = 3, b: float = 1.0,
                 clamp_range: Optional[Tuple[float, float]] = (0.0, 1.0),
                 conv2_from_x1: bool = False):
        super().__init__()
        self.k_est = KEstimationAOD(in_ch, conv2_from_x1=conv2_from_x1)
        self.register_buffer("b", torch.tensor(float(b)))
        self.clamp_range = clamp_range

    def forward(self, x: torch.Tensor):
        K = self.k_est(x)
        if K.shape != x.shape:
            raise RuntimeError(f"Shape mismatch: K={K.shape}, x={x.shape}")
        J = K * x - K + self.b
        if self.clamp_range is not None:
            lo, hi = self.clamp_range
            J = torch.clamp(J, lo, hi)
        return J


# ---------- Core AOD + PONO/MS ----------
class AODPONOCore(nn.Module):
    """
    PONO at early feats, MS to re-inject moments before deeper convs,
    then standard K-estimation head and paper formula.
    """
    def __init__(self, in_ch: int = 3, b: float = 1.0,
                 clamp_range: Optional[Tuple[float, float]] = (0.0, 1.0),
                 conv2_from_x1: bool = False):
        super().__init__()
        C = in_ch
        # 前两层与 K-estimation 对齐，但加入 PONO / MS
        self.c1 = nn.Conv2d(C, 3, 1, padding=0, bias=True)
        self.c2 = nn.Conv2d(C, 3, 3, padding=1, bias=True)
        self.pono = PONO(affine=False)
        self.ms = MS()

        self.c3 = nn.Conv2d(6, 3, 5, padding=2, bias=True)
        self.c4 = nn.Conv2d(6, 3, 7, padding=3, bias=True)
        self.c5 = nn.Conv2d(12, C, 3, padding=1, bias=True)

        self.register_buffer("b", torch.tensor(float(b)))
        self.clamp_range = clamp_range
        self.conv2_from_x1 = bool(conv2_from_x1)

    def forward(self, x: torch.Tensor):
        x1 = F.relu(self.c1(x))
        # 与仓库相近：有的版本在 cat1 前后插 PONO；这里在 x1/x2 上 PONO，并在后续用 MS 注回
        x2_in = x1 if self.conv2_from_x1 else x
        x2 = F.relu(self.c2(x2_in))

        cat1 = torch.cat((x1, x2), 1)

        # PONO 提取位置统计
        x1_n, mean1, std1 = self.pono(x1)
        x2_n, mean2, std2 = self.pono(x2)

        x3 = F.relu(self.c3(cat1))
        cat2 = torch.cat((x2_n, x3), 1)   # 与部分实现一致：用归一化后的 x2_n 融合

        # 将 (mean1, std1) 注回 x3，(mean2, std2) 注回后续特征
        x3 = self.ms(x3, mean1, std1)
        x4 = F.relu(self.c4(cat2))
        x4 = self.ms(x4, mean2, std2)

        cat3 = torch.cat((x1_n, x2_n, x3, x4), 1)
        K = self.c5(cat3)  # linear head for K

        if K.shape != x.shape:
            raise RuntimeError(f"Shape mismatch: K={K.shape}, x={x.shape}")

        J = K * x - K + self.b
        if self.clamp_range is not None:
            lo, hi = self.clamp_range
            J = torch.clamp(J, lo, hi)
        return J


# ---------- Ultralytics-friendly wrappers ----------
class AODNetPaperULY(nn.Module):
    """
    Ultralytics parser signature: (c1, c2=None, *args)
    Args (after c2):
      b: float = 1.0
      clamp_lo: float = 0.0
      clamp_hi: float = 1.0
      detach: bool = False        (detach input during training)
      conv2_from_x1: bool = False (True to mimic some GitHub variants)
    """
    def __init__(self, c1: int, c2: Optional[int] = None,
                 b: float = 1.0, clamp_lo: float = 0.0, clamp_hi: float = 1.0,
                 detach: bool = False, conv2_from_x1: bool = False):
        super().__init__()
        self.c1 = int(c1)
        self.c2 = self.c1
        self.detach = bool(detach)
        self.core = AODNetPaperCore(
            in_ch=self.c1, b=b,
            clamp_range=(clamp_lo, clamp_hi),
            conv2_from_x1=conv2_from_x1
        )

    def forward(self, x: torch.Tensor):
        x_in = x.detach() if (self.detach and self.training) else x
        return self.core(x_in)


class AODPONONetULY(nn.Module):
    """
    Ultralytics parser signature: (c1, c2=None, *args)
    Args (after c2):
      b: float = 1.0
      clamp_lo: float = 0.0
      clamp_hi: float = 1.0
      detach: bool = False
      conv2_from_x1: bool = False
    """
    def __init__(self, c1: int, c2: Optional[int] = None,
                 b: float = 1.0, clamp_lo: float = 0.0, clamp_hi: float = 1.0,
                 detach: bool = False, conv2_from_x1: bool = False):
        super().__init__()
        self.c1 = int(c1)
        self.c2 = self.c1
        self.detach = bool(detach)
        self.core = AODPONOCore(
            in_ch=self.c1, b=b,
            clamp_range=(clamp_lo, clamp_hi),
            conv2_from_x1=conv2_from_x1
        )

    def forward(self, x: torch.Tensor):
        x_in = x.detach() if (self.detach and self.training) else x
        return self.core(x_in)
