import torch
import torch.nn as nn
from torchvision.models import resnet18
from dataclasses import dataclass
from typing import Union, Tuple, Optional, Dict

@dataclass
class RotPermConfig:
    n_classes: int = 10
    topo: Optional[Dict] = None

class Conv2d_rotk(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels_rot4: int,
        out_channels_rot2: int,
        out_channels_rot0: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        swap: bool = False
    ):
        super(Conv2d_rotk, self).__init__()

        self.in_channels = in_channels
        self.out_channels_rot4 = out_channels_rot4
        self.out_channels_rot2 = out_channels_rot2
        self.out_channels_rot0 = out_channels_rot0
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        cantk = self.out_channels_rot4 + self.out_channels_rot2 + self.out_channels_rot0
        self.kernels = nn.Parameter(torch.empty(cantk, self.in_channels, self.kernel_size[0], self.kernel_size[1]))
        nn.init.kaiming_normal_(self.kernels, mode='fan_out', nonlinearity='relu')
        
        if bias:
            out_channels_total = 4 * out_channels_rot4 + 2 * out_channels_rot2 + out_channels_rot0
            self.bias_param = nn.Parameter(torch.empty(out_channels_total))
            nn.init.zeros_(self.bias_param)
        else:
            self.register_parameter('bias_param', None)

        # Permutations
        if out_channels_rot2 > 0:
            smr2 = torch.zeros(in_channels, in_channels)
            if swap:
                for i in range(0, in_channels, 2):
                    if i + 1 < in_channels:
                        smr2[i, i+1] = 1.
                        smr2[i+1, i] = 1.
            else:
                smr2 = torch.eye(in_channels)
            self.register_buffer('smr2', smr2)

        if out_channels_rot4 > 0:
            smr4a = torch.zeros(in_channels, in_channels)
            smr4b = torch.zeros(in_channels, in_channels)
            smr4c = torch.zeros(in_channels, in_channels)
            if swap:
                for i in range(0, in_channels, 4):
                    if i + 3 < in_channels:
                        smr4a[i+0, i+1] = 1.
                        smr4a[i+1, i+2] = 1.
                        smr4a[i+2, i+3] = 1.
                        smr4a[i+3, i+0] = 1.
                        
                        smr4b[i+0, i+2] = 1.
                        smr4b[i+1, i+3] = 1.
                        smr4b[i+2, i+0] = 1.
                        smr4b[i+3, i+1] = 1.
                        
                        smr4c[i+0, i+3] = 1.
                        smr4c[i+1, i+0] = 1.
                        smr4c[i+2, i+1] = 1.
                        smr4c[i+3, i+2] = 1.
            else:
                smr4a = torch.eye(in_channels)
                smr4b = smr4a
                smr4c = smr4a
            self.register_buffer('smr4a', smr4a)
            self.register_buffer('smr4b', smr4b)
            self.register_buffer('smr4c', smr4c)

    def create_convkernels(self):
        res = []
        idx = 0
        if self.out_channels_rot4 > 0:
            k4 = self.kernels[idx:idx+self.out_channels_rot4]
            idx += self.out_channels_rot4
            k4_0 = k4
            k4_1 = torch.rot90(torch.einsum('ij,njhw->nihw', self.smr4a, k4), k=1, dims=[2, 3])
            k4_2 = torch.rot90(torch.einsum('ij,njhw->nihw', self.smr4b, k4), k=2, dims=[2, 3])
            k4_3 = torch.rot90(torch.einsum('ij,njhw->nihw', self.smr4c, k4), k=3, dims=[2, 3])
            # Interleave: (N4, 4, Cin, H, W)
            k4_all = torch.stack([k4_0, k4_1, k4_2, k4_3], dim=1).flatten(0, 1)
            res.append(k4_all)
            
        if self.out_channels_rot2 > 0:
            k2 = self.kernels[idx:idx+self.out_channels_rot2]
            idx += self.out_channels_rot2
            k2_0 = k2
            k2_1 = torch.rot90(torch.einsum('ij,njhw->nihw', self.smr2, k2), k=1, dims=[2, 3])
            k2_all = torch.stack([k2_0, k2_1], dim=1).flatten(0, 1)
            res.append(k2_all)
            
        if self.out_channels_rot0 > 0:
            k0 = self.kernels[idx:idx+self.out_channels_rot0]
            res.append(k0)
            
        return torch.cat(res, dim=0)

    def forward(self, x):
        return torch.nn.functional.conv2d(
            x, self.create_convkernels(), 
            bias=self.bias_param, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation, 
            groups=self.groups
        )

class RotationPool(nn.Module):
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size
        
    def forward(self, x):
        if self.group_size <= 1:
            return x
        batch_size, channels = x.shape
        # Ensure channels is a multiple of group_size
        if channels % self.group_size != 0:
            return x # Fallback
        x = x.view(batch_size, channels // self.group_size, self.group_size)
        return x.mean(dim=2)

class RotPermResNet18(nn.Module):
    def __init__(self, topo, n_classes=10):
        super().__init__()
        self.model = resnet18(weights=None)
        
        # Replace layers
        # conv1
        c1 = topo['conv1']
        ch0 = 4*c1['r4'] + 2*c1['r2'] + c1['r0']
        self.model.conv1 = Conv2d_rotk(3, c1['r4'], c1['r2'], c1['r0'], 7, stride=2, padding=3, bias=False, swap=False)
        self.model.bn1 = nn.BatchNorm2d(ch0)
        
        # layer1
        m1 = topo['mod1']
        ch1 = 4*m1['r4'] + 2*m1['r2'] + m1['r0']
        self._replace_resnet_layer(self.model.layer1, ch0, m1)
        
        # layer2
        m2 = topo['mod2']
        ch2 = 4*m2['r4'] + 2*m2['r2'] + m2['r0']
        self._replace_resnet_layer(self.model.layer2, ch1, m2, stride=2)
        
        # layer3
        m3 = topo['mod3']
        ch3 = 4*m3['r4'] + 2*m3['r2'] + m3['r0']
        self._replace_resnet_layer(self.model.layer3, ch2, m3, stride=2)
        
        # layer4
        m4 = topo['mod4']
        ch4 = 4*m4['r4'] + 2*m4['r2'] + m4['r0']
        self._replace_resnet_layer(self.model.layer4, ch3, m4, stride=2)
        
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Assuming we mostly use r4 in the last layer for pooling
        self.rotation_pool = RotationPool(group_size=4 if m4['r4'] > 0 else 1)
        self.model.fc = nn.Linear(ch4 // (4 if m4['r4'] > 0 else 1), n_classes)

    def _replace_resnet_layer(self, layer, in_channels, mod_topo, stride=1):
        r4, r2, r0 = mod_topo['r4'], mod_topo['r2'], mod_topo['r0']
        swap = mod_topo.get('swap', True)
        out_channels = 4*r4 + 2*r2 + r0
        
        # Block 0
        layer[0].conv1 = Conv2d_rotk(in_channels, r4, r2, r0, 3, stride=stride, padding=1, bias=False, swap=swap)
        layer[0].bn1 = nn.BatchNorm2d(out_channels)
        layer[0].conv2 = Conv2d_rotk(out_channels, r4, r2, r0, 3, stride=1, padding=1, bias=False, swap=swap)
        layer[0].bn2 = nn.BatchNorm2d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            layer[0].downsample = nn.Sequential(
                Conv2d_rotk(in_channels, r4, r2, r0, 1, stride=stride, bias=False, swap=swap),
                nn.BatchNorm2d(out_channels)
            )
            
        # Block 1
        layer[1].conv1 = Conv2d_rotk(out_channels, r4, r2, r0, 3, stride=1, padding=1, bias=False, swap=swap)
        layer[1].bn1 = nn.BatchNorm2d(out_channels)
        layer[1].conv2 = Conv2d_rotk(out_channels, r4, r2, r0, 3, stride=1, padding=1, bias=False, swap=swap)
        layer[1].bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.rotation_pool(x)
        x = self.model.fc(x)
        return x

def build(config: RotPermConfig):
    if config.topo is None:
        # Default topo matching resnet18's channel growth but with rotperm
        # r4: 16 -> 64 channels
        # r4: 32 -> 128 channels
        # r4: 64 -> 256 channels
        # r4: 128 -> 512 channels
        config.topo = {
            'conv1': {'r4': 16,   'r2': 0,   'r0': 0,                },
            'mod1':  {'r4': 16,   'r2': 0,   'r0': 0,   "swap": True },
            'mod2':  {'r4': 32,   'r2': 0,   'r0': 0,   "swap": True },
            'mod3':  {'r4': 64,   'r2': 0,   'r0': 0,   "swap": True },
            'mod4':  {'r4': 128,  'r2': 0,   'r0': 0,   "swap": True }
        }
    return RotPermResNet18(config.topo, n_classes=config.n_classes)
