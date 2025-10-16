# parallel_dino.py — Parallel DINO backbone with multiple fusion modes (concat/sum/adaptive/spatial/dynamic/multiscale)
from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn
import timm

# Reuse Ultralytics blocks for consistent behavior
from ultralytics.nn.modules import Conv, C3k2, A2C2f


# ---------- small helpers ----------
class _ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1, p: int = 0, act: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=False), nn.BatchNorm2d(out_ch)]
        if act:
            layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Take(nn.Module):
    """Return one element from a list by index and expose channels to the YOLO parser."""
    def __init__(self, index: int, out_ch: int | None = None):
        super().__init__()
        self.index = int(index)
        self.out_channels = int(out_ch) if out_ch is not None else None
        self.c2 = int(out_ch) if out_ch is not None else None

    def forward(self, x):
        y = x[self.index] if isinstance(x, (list, tuple)) else x
        if self.out_channels is None and hasattr(y, "shape") and len(y.shape) > 1:
            ch = int(y.shape[1])
            self.out_channels = ch
            self.c2 = ch
        return y


class Scale(nn.Module):
    """Learnable scalar gate: out = alpha * x"""
    def __init__(self, init: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.alpha


class _DWConv3x3(nn.Module):
    """Depthwise 3x3 + BN + SiLU, then pointwise 1x1 + BN + SiLU."""
    def __init__(self, ch: int, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, padding=dilation, dilation=dilation, groups=ch, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------- fusion methods ----------
class AdaptiveFusion(nn.Module):
    """SE attention on DINO + 1x1 reduction -> concat -> mixing"""
    def __init__(self, yolo_ch: int, dino_ch: int, out_ch: int, reduction_ratio: int = 4):
        super().__init__()
        reduced_ch = max(4, dino_ch // max(1, reduction_ratio))

        se_hidden = max(4, dino_ch // 16)
        self.dino_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dino_ch, se_hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(se_hidden, dino_ch, 1, bias=True),
            nn.Sigmoid(),
        )
        self.dino_reduce = _ConvBNAct(dino_ch, reduced_ch, k=1, act=True)

        self.fuse_conv = nn.Sequential(
            _ConvBNAct(yolo_ch + reduced_ch, out_ch, k=1, act=True),
            _DWConv3x3(out_ch, dilation=1),
        )

    def forward(self, yolo_feat: torch.Tensor, dino_feat: torch.Tensor) -> torch.Tensor:
        att = self.dino_se(dino_feat)
        dino_weighted = dino_feat * att
        dino_reduced = self.dino_reduce(dino_weighted)
        fused = torch.cat([yolo_feat, dino_reduced], dim=1)
        return self.fuse_conv(fused)


class SpatialGuidedFusion(nn.Module):
    """Spatial-guided fusion: large-kernel attention from YOLO guides gated DINO, then DW mixing"""
    def __init__(self, channels: int, attention_kernel: int = 11, dilation: int = 2):
        super().__init__()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=attention_kernel, padding=attention_kernel // 2, bias=False),
            nn.Sigmoid()
        )
        self.dw_conv = _DWConv3x3(channels, dilation=dilation)
        self.dino_gate = Scale(init=0.5)

    def forward(self, yolo_feat: torch.Tensor, dino_feat: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(yolo_feat, dim=1, keepdim=True)
        max_out, _ = torch.max(yolo_feat, dim=1, keepdim=True)
        spatial_weights = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        weighted_dino = self.dino_gate(dino_feat) * spatial_weights
        fused = yolo_feat + weighted_dino
        return self.dw_conv(fused)


class DynamicWeightedFusion(nn.Module):
    """Dynamic channel-wise weights from global context of both branches"""
    def __init__(self, channels: int):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(16, channels // 4)
        self.weight_net = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )
        self.dw_conv = _DWConv3x3(channels, dilation=1)

    def forward(self, yolo_feat: torch.Tensor, dino_feat: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = yolo_feat.shape
        yolo_global = self.global_pool(yolo_feat).view(B, C)
        dino_global = self.global_pool(dino_feat).view(B, C)
        combined = torch.cat([yolo_global, dino_global], dim=1)
        weights = self.weight_net(combined).view(B, C, 1, 1)
        fused = yolo_feat + weights * dino_feat
        return self.dw_conv(fused)


class MultiScaleSpatialFusion(nn.Module):
    """Multi-scale: different receptive fields by pyramid level."""
    def __init__(self, channels: int, level: str = "p3"):
        super().__init__()
        if level == "p3":
            attention_kernel, dilation = 7, 1
        elif level == "p4":
            attention_kernel, dilation = 9, 1
        else:  # p5
            attention_kernel, dilation = 11, 2
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=attention_kernel, padding=attention_kernel // 2, bias=False),
            nn.Sigmoid()
        )
        self.dw_conv = _DWConv3x3(channels, dilation=dilation)
        self.dino_gate = Scale(init=0.5)

    def forward(self, yolo_feat: torch.Tensor, dino_feat: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(yolo_feat, dim=1, keepdim=True)
        max_out, _ = torch.max(yolo_feat, dim=1, keepdim=True)
        spatial_weights = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        weighted_dino = self.dino_gate(dino_feat) * spatial_weights
        fused = yolo_feat + weighted_dino
        return self.dw_conv(fused)


# ---------- encoders ----------
class DINOEncoder(nn.Module):
    """
    DINO(v2/v3/convnext_dino*) encoder with internal 1x1 projections to (256,512,1024).
    """
    def __init__(
        self,
        model_name: str = "convnext_small.dinov3_lvd1689m",
        pretrained: bool = True,
        freeze: bool = True,
        use_imagenet_norm: bool = False,
        proj_out: Tuple[int, int, int] = (256, 512, 1024),
    ):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        enc_ch = self.encoder.feature_info.channels()
        assert len(enc_ch) >= 3, f"{model_name} exposes only {len(enc_ch)} features; need >=3."
        self.pick = (-3, -2, -1)
        c3_ch, c4_ch, c5_ch = enc_ch[self.pick[0]], enc_ch[self.pick[1]], enc_ch[self.pick[2]]

        self.proj3 = _ConvBNAct(c3_ch, proj_out[0], k=1, act=True)
        self.proj4 = _ConvBNAct(c4_ch, proj_out[1], k=1, act=True)
        self.proj5 = _ConvBNAct(c5_ch, proj_out[2], k=1, act=True)

        self.out_channels = list(proj_out)
        self.use_imagenet_norm = use_imagenet_norm
        if use_imagenet_norm:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer("img_mean", mean, persistent=False)
            self.register_buffer("img_std", std, persistent=False)

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.use_imagenet_norm:
            x = (x - self.img_mean) / self.img_std
        feats = self.encoder(x)
        c3, c4, c5 = feats[self.pick[0]], feats[self.pick[1]], feats[self.pick[2]]
        return [self.proj3(c3), self.proj4(c4), self.proj5(c5)]


class DINOYolo12Backbone(nn.Module):
    """
    Enhanced YOLOv12 backbone with DINO parallel branch.
    Returns [P3, P4, P5] with channels [256, 512, 1024].
    Fusion modes: "concat", "sum", "adaptive", "spatial", "dynamic", "multiscale".
    """
    def __init__(
        self,
        dino_name: str = "convnext_small.dinov3_lvd1689m",
        dino_pretrained: bool = True,
        dino_freeze: bool = True,
        dino_imagenet_norm: bool = False,
        fuse_mode: str = "concat",
        attention_kernel: int = 11,
    ):
        super().__init__()

        # ---- YOLO12 trunk ----
        self.stem1 = Conv(3, 64, 3, 2)            # P1/2
        self.stem2 = Conv(64, 128, 3, 2)          # P2/4
        self.c3 = C3k2(128, 256, 2, False, 0.25)
        self.p3 = Conv(256, 256, 3, 2)            # P3/8
        self.c4 = C3k2(256, 512, 2, False, 0.25)
        self.p4 = Conv(512, 512, 3, 2)            # P4/16
        self.c5 = A2C2f(512, 512, 4, True, 4)
        self.p5 = Conv(512, 1024, 3, 2)           # P5/32
        self.c5b = A2C2f(1024, 1024, 4, True, 1)

        # ---- DINO branch ----
        self.dino = DINOEncoder(
            model_name=dino_name,
            pretrained=dino_pretrained,
            freeze=dino_freeze,
            use_imagenet_norm=dino_imagenet_norm,
            proj_out=(256, 512, 1024),
        )

        # ---- Fusion ----
        self.fuse_mode = fuse_mode
        if fuse_mode == "concat":
            self.fuse3 = nn.Sequential(_ConvBNAct(256 + 256, 256, k=1, act=True), _DWConv3x3(256, dilation=1))
            self.fuse4 = nn.Sequential(_ConvBNAct(512 + 512, 512, k=1, act=True), _DWConv3x3(512, dilation=1))
            self.fuse5 = nn.Sequential(_ConvBNAct(1024 + 1024, 1024, k=1, act=True), _DWConv3x3(1024, dilation=2))
        elif fuse_mode == "sum":
            self.d3_gate = Scale(0.5); self.fuse3 = _DWConv3x3(256, dilation=1)
            self.d4_gate = Scale(0.5); self.fuse4 = _DWConv3x3(512, dilation=1)
            self.d5_gate = Scale(0.5); self.fuse5 = _DWConv3x3(1024, dilation=2)
        elif fuse_mode == "adaptive":
            self.fuse3 = AdaptiveFusion(256, 256, 256, reduction_ratio=4)
            self.fuse4 = AdaptiveFusion(512, 512, 512, reduction_ratio=4)
            self.fuse5 = AdaptiveFusion(1024, 1024, 1024, reduction_ratio=4)
        elif fuse_mode == "spatial":
            self.fuse3 = SpatialGuidedFusion(256, attention_kernel=attention_kernel, dilation=1)
            self.fuse4 = SpatialGuidedFusion(512, attention_kernel=attention_kernel, dilation=1)
            self.fuse5 = SpatialGuidedFusion(1024, attention_kernel=attention_kernel, dilation=2)
        elif fuse_mode == "dynamic":
            self.fuse3 = DynamicWeightedFusion(256)
            self.fuse4 = DynamicWeightedFusion(512)
            self.fuse5 = DynamicWeightedFusion(1024)
        elif fuse_mode == "multiscale":
            self.fuse3 = MultiScaleSpatialFusion(256, level="p3")
            self.fuse4 = MultiScaleSpatialFusion(512, level="p4")
            self.fuse5 = MultiScaleSpatialFusion(1024, level="p5")
        else:
            raise ValueError("fuse_mode must be one of: 'concat', 'sum', 'adaptive', 'spatial', 'dynamic', 'multiscale'")

        self.out_channels = [256, 512, 1024]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # YOLO12 trunk
        x1 = self.stem1(x)
        x2 = self.stem2(x1)
        x3 = self.c3(x2)
        p3 = self.p3(x3)

        x4 = self.c4(p3)
        p4 = self.p4(x4)

        x5 = self.c5(p4)
        p5 = self.p5(x5)
        y5 = self.c5b(p5)

        # DINO parallel
        d3, d4, d5 = self.dino(x)

        # Fusion
        if self.fuse_mode == "concat":
            f3 = self.fuse3(torch.cat([p3, d3], dim=1))
            f4 = self.fuse4(torch.cat([p4, d4], dim=1))
            f5 = self.fuse5(torch.cat([y5, d5], dim=1))
        elif self.fuse_mode == "sum":
            f3 = self.fuse3(p3 + self.d3_gate(d3))
            f4 = self.fuse4(p4 + self.d4_gate(d4))
            f5 = self.fuse5(y5 + self.d5_gate(d5))
        else:  # adaptive / spatial / dynamic / multiscale
            f3 = self.fuse3(p3, d3)
            f4 = self.fuse4(p4, d4)
            f5 = self.fuse5(y5, d5)

        return [f3, f4, f5]  # P3/8, P4/16, P5/32


# quick sanity test
if __name__ == "__main__":
    x = torch.randn(2, 3, 640, 640)
    for mode in ["concat", "sum", "adaptive", "spatial", "dynamic", "multiscale"]:
        backbone = DINOYolo12Backbone(dino_name="convnext_small.dinov3_lvd1689m", fuse_mode=mode)
        outs = backbone(x)
        print(mode, "->", [tuple(o.shape) for o in outs])
