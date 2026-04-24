import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from typing import Union, Tuple

_size_2_t = Union[float, Tuple[float, float]]

class Conv2d_rotk(nn.Module):
    def __init__(self):
        super(Conv2d_rotk, self).__init__()
        self.fn = Conv2d_rotk.apply
        self.weight = nn.Parameter(torch.randn(1, 1, 2, 2))

    def __init__(
        self,
        in_channels: int,
        out_channels_rot4: int, # orig + 3 rot 90º
        out_channels_rot2: int, # orig + 1 rot 90º
        out_channels_rot0: int, # orig (no hay rotación)
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        swap=False
    ):
        super(Conv2d_rotk, self).__init__()

        self.in_channels = in_channels
        self.out_channels_rot4 = out_channels_rot4
        self.out_channels_rot2 = out_channels_rot2
        self.out_channels_rot0 = out_channels_rot0
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.device = device
        self.dtype = dtype

        self.fn = Conv2d_rotk.apply

        cantk = self.out_channels_rot4 + self.out_channels_rot2 + self.out_channels_rot0
        self.kernels = torch.nn.Parameter(torch.empty(cantk, self.in_channels, self.kernel_size[0], self.kernel_size[1]), requires_grad=True)
        nn.init.kaiming_normal_(self.kernels, mode='fan_out', nonlinearity='relu')


# Permutaciones
        if out_channels_rot2 > 0:
          if swap:
            self.smr2 = torch.zeros(in_channels, in_channels, dtype=self.kernels.dtype).to(device)
            for i in range(0, in_channels, 2):
              self.smr2[i, i+1] = 1.
              self.smr2[i+1, i] = 1.
          else:
            self.smr2 = torch.eye(in_channels, dtype=self.kernels.dtype).to(device) # swap matrix para r2
            self.smr2[0,0] = 1.0 # solo para realizar las mismas operaciones que en el otro caso

        if out_channels_rot4 > 0:
          if swap:
            self.smr4a = torch.zeros(in_channels, in_channels, dtype=self.kernels.dtype).to(device)
            self.smr4b = torch.zeros(in_channels, in_channels, dtype=self.kernels.dtype).to(device)
            self.smr4c = torch.zeros(in_channels, in_channels, dtype=self.kernels.dtype).to(device)
            for i in range(0, in_channels, 4):
              self.smr4a[i+0, i+1] = 1.
              self.smr4a[i+1, i+2] = 1.
              self.smr4a[i+2, i+3] = 1.
              self.smr4a[i+3, i+0] = 1.
              self.smr4b[i+0, i+2] = 1.
              self.smr4b[i+1, i+3] = 1.
              self.smr4b[i+2, i+0] = 1.
              self.smr4b[i+3, i+1] = 1.
              self.smr4c[i+0, i+3] = 1.
              self.smr4c[i+1, i+0] = 1.
              self.smr4c[i+2, i+1] = 1.
              self.smr4c[i+3, i+2] = 1.
          else:
            self.smr4a = torch.eye(in_channels, dtype=self.kernels.dtype).to(device) # swap matrix for r2
            self.smr4a[0,0] = 1.0 # solo para realizar las mismas operaciones que en el otro caso
            self.smr4b = self.smr4c = self.smr4a

    def swap_r2(self,kernels):
        return(torch.permute(torch.matmul(self.smr2, torch.permute(kernels, (0,2,1,3))), (0,2,1,3)))
    def swap_r4a(self,kernels):
        return(torch.permute(torch.matmul(self.smr4a, torch.permute(kernels, (0,2,1,3))), (0,2,1,3)))
    def swap_r4b(self,kernels):
        return(torch.permute(torch.matmul(self.smr4b, torch.permute(kernels, (0,2,1,3))), (0,2,1,3)))
    def swap_r4c(self,kernels):
        return(torch.permute(torch.matmul(self.smr4c, torch.permute(kernels, (0,2,1,3))), (0,2,1,3)))


    def create_convkernels(self):
        if self.out_channels_rot4 > 0:
          idx = 0
          convkernels = torch.cat((torch.rot90(self.kernels[idx:idx+1,:,:,:], k=0, dims=[2, 3]),
                                   torch.rot90(self.swap_r4a(self.kernels[idx:idx+1,:,:,:]), k=1, dims=[2, 3]),
                                   torch.rot90(self.swap_r4b(self.kernels[idx:idx+1,:,:,:]), k=2, dims=[2, 3]),
                                   torch.rot90(self.swap_r4c(self.kernels[idx:idx+1,:,:,:]), k=3, dims=[2, 3])))
          for i in range(self.out_channels_rot4 - 1):
            idx = idx + 1
            convkernels = torch.cat((convkernels,
                                     torch.rot90(self.kernels[idx:idx+1,:,:,:], k=0, dims=[2, 3]),
                                     torch.rot90(self.swap_r4a(self.kernels[idx:idx+1,:,:,:]), k=1, dims=[2, 3]),
                                     torch.rot90(self.swap_r4b(self.kernels[idx:idx+1,:,:,:]), k=2, dims=[2, 3]),
                                     torch.rot90(self.swap_r4c(self.kernels[idx:idx+1,:,:,:]), k=3, dims=[2, 3])))
          if self.out_channels_rot2 > 0:
            for i in range(self.out_channels_rot2):
              idx = idx + 1
              convkernels = torch.cat((convkernels,
                                       torch.rot90(self.kernels[idx:idx+1,:,:,:], k=0, dims=[2, 3]),
                                       torch.rot90(self.swap_r2(self.kernels[idx:idx+1,:,:,:]), k=1, dims=[2, 3])))
          if self.out_channels_rot0 > 0:
            for i in range(self.out_channels_rot0):
              idx = idx + 1
              convkernels = torch.cat((convkernels,
                                       self.kernels[idx:idx+1,:,:,:]))
          return convkernels
        else:
          if self.out_channels_rot2 > 0:
            idx = 0
            convkernels = torch.cat((torch.rot90(self.kernels[idx:idx+1,:,:,:], k=0, dims=[2, 3]),
                                     torch.rot90(self.swap_r2(self.kernels[idx:idx+1,:,:,:]), k=1, dims=[2, 3])))
            for i in range(self.out_channels_rot2):
              idx = idx + 1
              convkernels = torch.cat((convkernels,
                                      torch.rot90(self.kernels[idx:idx+1,:,:,:], k=0, dims=[2, 3]),
                                      torch.rot90(self.swap_r2(self.kernels[idx:idx+1,:,:,:]), k=1, dims=[2, 3])))
            if self.out_channels_rot0 > 0:
              for i in range(self.out_channels_rot0):
                idx = idx + 1
                convkernels = torch.cat((convkernels,
                                         self.kernels[idx:idx+1,:,:,:]))
            return convkernels
          else:
            return self.kernels

    def forward(self, x):
        x =  torch.nn.functional.conv2d(x, self.create_convkernels(), bias=self.bias, stride=self.stride, padding=self.padding)
        return x

class NeuralNetwork(nn.Module):
    def __init__(
        self,
        topo: dict,
        device: str = None
    ):
        super().__init__()
        self.topo = topo
        self.orig_model = resnet18(weights=None)

        # primera capa de convolución
        r4 = self.topo['conv1']['r4']
        r2 = self.topo['conv1']['r2']
        r0 = self.topo['conv1']['r0']
        ch0 = 4*r4 + 2*r2 + r0
        self.orig_model._modules['conv1'] = Conv2d_rotk(3, r4, r2, r0, (7, 7), bias=None, stride=(2, 2), padding=(3, 3), swap=False, device=device)
        self.orig_model._modules['bn1'] = nn.BatchNorm2d(ch0, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # primer bloque ResNet ("layer1")
        r4 = self.topo['mod1']['r4']
        r2 = self.topo['mod1']['r2']
        r0 = self.topo['mod1']['r0']
        ch1 = 4*r4 + 2*r2 + r0
        swap = self.topo['mod1']['swap']
        self.orig_model._modules['layer1'][0]._modules['conv1'] = Conv2d_rotk(ch0, r4, r2, r0, (3, 3), bias=None, stride=(1, 1), padding=(1, 1), swap=swap, device=device)
        self.orig_model._modules['layer1'][0]._modules['conv2'] = Conv2d_rotk(ch1, r4, r2, r0, (3, 3), bias=None, stride=(1, 1), padding=(1, 1), swap=swap, device=device)
        self.orig_model._modules['layer1'][1]._modules['conv1'] = Conv2d_rotk(ch1, r4, r2, r0, (3, 3), bias=None, stride=(1, 1), padding=(1, 1), swap=swap, device=device)
        self.orig_model._modules['layer1'][1]._modules['conv2'] = Conv2d_rotk(ch1, r4, r2, r0, (3, 3), bias=None, stride=(1, 1), padding=(1, 1), swap=swap, device=device)
        self.orig_model._modules['layer1'][0]._modules['bn1'] = nn.BatchNorm2d(ch1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.orig_model._modules['layer1'][0]._modules['bn2'] = nn.BatchNorm2d(ch1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.orig_model._modules['layer1'][1]._modules['bn1'] = nn.BatchNorm2d(ch1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.orig_model._modules['layer1'][1]._modules['bn2'] = nn.BatchNorm2d(ch1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # segundo bloque ResNet ("layer2")
        r4 = self.topo['mod2']['r4']
        r2 = self.topo['mod2']['r2']
        r0 = self.topo['mod2']['r0']
        ch2 = 4*r4 + 2*r2 + r0
        swap = self.topo['mod2']['swap']
        self.orig_model._modules['layer2'][0]._modules['conv1'] =         Conv2d_rotk(ch1, r4, r2, r0, (3, 3), bias=None, stride=(2, 2), padding=(1, 1), swap=swap, device=device)
        self.orig_model._modules['layer2'][0]._modules['conv2'] =         Conv2d_rotk(ch2, r4, r2, r0, (3, 3), bias=None, stride=(1, 1), padding=(1, 1), swap=swap, device=device)
        self.orig_model._modules['layer2'][0]._modules['downsample'][0] = Conv2d_rotk(ch1, r4, r2, r0, (1, 1), bias=None, stride=(2, 2), padding=(0, 0), swap=swap, device=device)
        self.orig_model._modules['layer2'][1]._modules['conv1'] =         Conv2d_rotk(ch2, r4, r2, r0, (3, 3), bias=None, stride=(1, 1), padding=(1, 1), swap=swap, device=device)
        self.orig_model._modules['layer2'][1]._modules['conv2'] =         Conv2d_rotk(ch2, r4, r2, r0, (3, 3), bias=None, stride=(1, 1), padding=(1, 1), swap=swap, device=device)
        self.orig_model._modules['layer2'][0]._modules['bn1'] =           nn.BatchNorm2d(ch2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.orig_model._modules['layer2'][0]._modules['bn2'] =           nn.BatchNorm2d(ch2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.orig_model._modules['layer2'][0]._modules['downsample'][1] = nn.BatchNorm2d(ch2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.orig_model._modules['layer2'][1]._modules['bn1'] =           nn.BatchNorm2d(ch2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.orig_model._modules['layer2'][1]._modules['bn2'] =           nn.BatchNorm2d(ch2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # tercer bloque ResNet ("layer3")
        r4 = self.topo['mod3']['r4']
        r2 = self.topo['mod3']['r2']
        r0 = self.topo['mod3']['r0']
        ch3 = 4*r4 + 2*r2 + r0
        swap = self.topo['mod3']['swap']
        self.orig_model._modules['layer3'][0]._modules['conv1'] =         Conv2d_rotk(ch2, r4, r2, r0, (3, 3), bias=None, stride=(2, 2), padding=(1, 1), swap=swap, device=device)
        self.orig_model._modules['layer3'][0]._modules['conv2'] =         Conv2d_rotk(ch3, r4, r2, r0, (3, 3), bias=None, stride=(1, 1), padding=(1, 1), swap=swap, device=device)
        self.orig_model._modules['layer3'][0]._modules['downsample'][0] = Conv2d_rotk(ch2, r4, r2, r0, (1, 1), bias=None, stride=(2, 2), padding=(0, 0), swap=swap, device=device)
        self.orig_model._modules['layer3'][1]._modules['conv1'] =         Conv2d_rotk(ch3, r4, r2, r0, (3, 3), bias=None, stride=(1, 1), padding=(1, 1), swap=swap, device=device)
        self.orig_model._modules['layer3'][1]._modules['conv2'] =         Conv2d_rotk(ch3, r4, r2, r0, (3, 3), bias=None, stride=(1, 1), padding=(1, 1), swap=swap, device=device)
        self.orig_model._modules['layer3'][0]._modules['bn1'] =           nn.BatchNorm2d(ch3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.orig_model._modules['layer3'][0]._modules['bn2'] =           nn.BatchNorm2d(ch3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.orig_model._modules['layer3'][0]._modules['downsample'][1] = nn.BatchNorm2d(ch3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.orig_model._modules['layer3'][1]._modules['bn1'] =           nn.BatchNorm2d(ch3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.orig_model._modules['layer3'][1]._modules['bn2'] =           nn.BatchNorm2d(ch3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # cuarto bloque ResNet ("layer4")
        r4 = self.topo['mod4']['r4']
        r2 = self.topo['mod4']['r2']
        r0 = self.topo['mod4']['r0']
        ch4 = 4*r4 + 2*r2 + r0
        swap = self.topo['mod4']['swap']
        self.orig_model._modules['layer4'][0]._modules['conv1'] =         Conv2d_rotk(ch3, r4, r2, r0, (3, 3), bias=None, stride=(2, 2), padding=(1, 1), swap=swap, device=device)
        self.orig_model._modules['layer4'][0]._modules['conv2'] =         Conv2d_rotk(ch4, r4, r2, r0, (3, 3), bias=None, stride=(1, 1), padding=(1, 1), swap=swap, device=device)
        self.orig_model._modules['layer4'][0]._modules['downsample'][0] = Conv2d_rotk(ch3, r4, r2, r0, (1, 1), bias=None, stride=(2, 2), padding=(0, 0), swap=swap, device=device)
        self.orig_model._modules['layer4'][1]._modules['conv1'] =         Conv2d_rotk(ch4, r4, r2, r0, (3, 3), bias=None, stride=(1, 1), padding=(1, 1), swap=swap, device=device)
        self.orig_model._modules['layer4'][1]._modules['conv2'] =         Conv2d_rotk(ch4, r4, r2, r0, (3, 3), bias=None, stride=(1, 1), padding=(1, 1), swap=swap, device=device)
        self.orig_model._modules['layer4'][0]._modules['bn1'] =           nn.BatchNorm2d(ch4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.orig_model._modules['layer4'][0]._modules['bn2'] =           nn.BatchNorm2d(ch4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.orig_model._modules['layer4'][0]._modules['downsample'][1] = nn.BatchNorm2d(ch4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.orig_model._modules['layer4'][1]._modules['bn1'] =           nn.BatchNorm2d(ch4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.orig_model._modules['layer4'][1]._modules['bn2'] =           nn.BatchNorm2d(ch4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.orig_model._modules['fc'] = nn.AvgPool1d((4), stride=(4))

        self.orig_model = nn.Sequential(self.orig_model, nn.Linear(in_features=int(ch4/4), out_features=1000, bias=True))

    def forward(self, x):
        return self.orig_model(x)
