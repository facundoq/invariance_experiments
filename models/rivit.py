import torch
import torch.nn as nn
import timm
from dataclasses import dataclass

@dataclass
class RI_ViTConfig:
    n_classes: int = 10
    num_rotations: int = 4
    model_name: str = 'vit_tiny_patch16_224'

class RotationInvariantViT(nn.Module):
    """
    Implementation of a Rotation Invariant Vision Transformer (RI-ViT).
    It processes multiple rotations of the input through a shared ViT backbone
    and pools the results to achieve invariance.
    """
    def __init__(self, config: RI_ViTConfig):
        super().__init__()
        self.num_rotations = config.num_rotations
        # Use a small ViT backbone from timm
        self.backbone = timm.create_model(
            config.model_name, 
            pretrained=False, 
            num_classes=0 # Feature extractor mode
        )
        self.head = nn.Linear(self.backbone.num_features, config.n_classes)

    def forward(self, x):
        # x shape: [batch, channels, h, w]
        batch_size = x.shape[0]
        
        # Generate rotated versions (lifting)
        rotated_xs = []
        for i in range(self.num_rotations):
            # Rotate by 90-degree increments using k=i
            rotated = torch.rot90(x, k=i, dims=(-2, -1))
            rotated_xs.append(rotated)
            
        # [batch * num_rotations, channels, h, w]
        stacked_x = torch.cat(rotated_xs, dim=0)
        
        # [batch * num_rotations, features]
        features = self.backbone(stacked_x)
        
        # Reshape to [num_rotations, batch, features]
        features = features.view(self.num_rotations, batch_size, -1)
        
        # Invariant Pooling: Mean across rotations
        # [batch, features]
        pooled_features = features.mean(dim=0)
        
        return self.head(pooled_features)

def build(config: RI_ViTConfig):
    return RotationInvariantViT(config)
