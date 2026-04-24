import torch
import torch.nn as nn
from torchvision.transforms.functional import rotate
from dataclasses import dataclass, field
from .resnet import build as build_resnet, ResNetConfig

@dataclass
class TIPoolingConfig:
    n_classes: int = 10
    num_rotations: int = 8
    base_config: ResNetConfig = field(default_factory=ResNetConfig)

class TIPoolingCNN(nn.Module):
    def __init__(self, base_model, num_rotations=8):
        super().__init__()
        self.base_model = base_model
        self.num_rotations = num_rotations

    def forward(self, x):
        batch_size = x.shape[0]
        # Generate rotated versions
        rotated_xs = [rotate(x, i * (360.0 / self.num_rotations)) for i in range(self.num_rotations)]
        stacked_x = torch.cat(rotated_xs, dim=0)
        
        features = self.base_model(stacked_x)
        
        # Reshape to [num_rotations, batch, ...]
        feat_shape = features.shape[1:]
        features = features.view(self.num_rotations, batch_size, *feat_shape)
        
        # Max pool across rotations
        pooled_features, _ = torch.max(features, dim=0)
        return pooled_features

def build(config: TIPoolingConfig):
    base_model = build_resnet(config.base_config)
    return TIPoolingCNN(base_model, num_rotations=config.num_rotations)
