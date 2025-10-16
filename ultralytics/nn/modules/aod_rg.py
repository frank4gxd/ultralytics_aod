import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# [Previous AOD code here - YOLOAODLayer, ResidualBlock, SEBlock, etc.]
# Copy the entire AOD module code from the previous artifact

class ResidualBlock(nn.Module):
    """Residual block with BatchNorm and configurable channels"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention block"""

    def __init__(self, channels: int, reduction: int = 4):
        super(SEBlock, self).__init__()
        reduced_channels = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced_channels, bias=False)
        self.fc2 = nn.Linear(reduced_channels, channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc2(self.relu(self.fc1(y)))
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class PONO(nn.Module):
    """Positional Normalization"""

    def __init__(self, input_size: Optional[tuple] = None, affine: bool = False, eps: float = 1e-5):
        super(PONO, self).__init__()
        self.eps = eps
        self.affine = affine

        if affine and input_size is not None:
            self.beta = nn.Parameter(torch.zeros(1, 1, *input_size))
            self.gamma = nn.Parameter(torch.ones(1, 1, *input_size))
        else:
            self.beta, self.gamma = None, None

    def forward(self, x: torch.Tensor) -> tuple:
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True) + self.eps).sqrt()
        x_norm = (x - mean) / std
        if self.affine and self.gamma is not None:
            x_norm = x_norm * self.gamma + self.beta
        return x_norm, mean, std


class MS(nn.Module):
    """Modulation and Scaling"""

    def __init__(self):
        super(MS, self).__init__()

    def forward(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean


class YOLOAODLayer(nn.Module):
    """YOLO-compatible AOD enhancement layer"""

    def __init__(self,
                 c1: int,
                 c2: Optional[int] = None,
                 use_attention: bool = True,
                 use_pono: bool = True,
                 lightweight: bool = False,
                 reduction: int = 4):
        super(YOLOAODLayer, self).__init__()

        self.c1 = c1
        self.c2 = c1  # Force channel preservation
        self.use_attention = use_attention
        self.use_pono = use_pono
        self.lightweight = lightweight

        # Channel configuration
        if lightweight:
            self.ch1 = min(16, c1)
            self.ch2 = min(24, c1 * 2)
        else:
            self.ch1 = min(32, c1 * 2)
            self.ch2 = min(48, c1 * 3)

        # Network layers
        self.conv1 = ResidualBlock(c1, self.ch1, kernel_size=1, padding=0)
        self.conv2 = ResidualBlock(self.ch1, self.ch1, kernel_size=3, padding=1)
        self.conv3 = ResidualBlock(self.ch1 * 2, self.ch2, kernel_size=5, padding=2)
        self.conv4 = ResidualBlock(self.ch1 + self.ch2, self.ch2, kernel_size=7, padding=3)

        total_channels = self.ch1 * 2 + self.ch2 * 2
        self.conv5 = ResidualBlock(total_channels, c1, kernel_size=3, padding=1)

        # Optional components
        if use_attention:
            self.att1 = SEBlock(self.ch1, reduction)
            self.att2 = SEBlock(self.ch1, reduction)
            self.att3 = SEBlock(self.ch2, reduction)
            self.att4 = SEBlock(self.ch2, reduction)
            self.att5 = SEBlock(c1, reduction)

        if use_pono:
            self.pono = PONO(affine=False)
            self.ms = MS()

        self.b = nn.Parameter(torch.ones(1))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1
        x1 = self.conv1(x)
        if self.use_attention:
            x1 = self.att1(x1)

        # Stage 2
        x2 = self.conv2(x1)
        if self.use_attention:
            x2 = self.att2(x2)

        cat1 = torch.cat((x1, x2), dim=1)

        # PONO normalization
        if self.use_pono:
            x1_norm, mean1, std1 = self.pono(x1)
            x2_norm, mean2, std2 = self.pono(x2)
        else:
            x1_norm, x2_norm = x1, x2
            mean1 = mean2 = std1 = std2 = None

        # Stage 3
        x3 = self.conv3(cat1)
        if self.use_attention:
            x3 = self.att3(x3)
        if self.use_pono and mean1 is not None:
            x3 = self.ms(x3, mean1, std1)

        # Stage 4
        cat2 = torch.cat((x2_norm, x3), dim=1)
        x4 = self.conv4(cat2)
        if self.use_attention:
            x4 = self.att4(x4)
        if self.use_pono and mean2 is not None:
            x4 = self.ms(x4, mean2, std2)

        # Stage 5
        cat3 = torch.cat((x1_norm, x2_norm, x3, x4), dim=1)
        k = self.conv5(cat3)
        if self.use_attention:
            k = self.att5(k)

        # Ensure spatial dimensions match
        if k.size() != x.size():
            k = F.interpolate(k, size=x.shape[2:], mode='bilinear', align_corners=False)

        # AOD enhancement formula
        output = k * x - k + self.b
        return F.relu(output)


class YOLOAODLightweight(YOLOAODLayer):
    """Lightweight AOD layer for mobile deployment"""

    def __init__(self, c1: int, c2: Optional[int] = None, use_attention: bool = False):
        super().__init__(
            c1=c1,
            c2=c2,
            use_attention=use_attention,
            use_pono=False,
            lightweight=True,
            reduction=8
        )


# Export aliases for easier access
AODLayer = YOLOAODLayer
AODLightweight = YOLOAODLightweight