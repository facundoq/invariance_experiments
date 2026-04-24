import torch
import torch.nn as nn
from escnn import nn as enn
from escnn import gspaces
from dataclasses import dataclass

@dataclass
class E2CNNConfig:
    n_classes: int = 10
    num_rotations: int = 8
    input_size: int = 224
    width_scale: int = 4 # Base multiplier for regular representations

class E2EquivariantCNN(nn.Module):
    def __init__(self, config: E2CNNConfig):
        super().__init__()
        
        # The cyclic group CN (N rotations)
        self.r2_space = gspaces.rot2dOnR2(N=config.num_rotations)
        
        # Input: 3 channels (RGB), trivial representation
        in_type = enn.FieldType(self.r2_space, 3 * [self.r2_space.trivial_repr])
        self.input_type = in_type
        
        # Intermediate: regular representation
        w = config.width_scale
        out_type = enn.FieldType(self.r2_space, w * [self.r2_space.regular_repr])
        
        self.block1 = enn.SequentialModule(
            enn.MaskModule(in_type, config.input_size, margin=1),
            enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type, inplace=True)
        )
        
        self.pool1 = enn.PointwiseMaxPool(out_type, kernel_size=2)
        
        # More channels
        out_type2 = enn.FieldType(self.r2_space, (w * 2) * [self.r2_space.regular_repr])
        self.block2 = enn.SequentialModule(
            enn.R2Conv(out_type, out_type2, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(out_type2),
            enn.ReLU(out_type2, inplace=True)
        )
        
        # Map to trivial representation (invariant)
        fc_dim = w * 16
        invariant_type = enn.FieldType(self.r2_space, fc_dim * [self.r2_space.trivial_repr])
        self.block3 = enn.SequentialModule(
            enn.R2Conv(out_type2, invariant_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(invariant_type),
            enn.ReLU(invariant_type, inplace=True)
        )
        
        self.fully_connected = nn.Linear(fc_dim, config.n_classes)

    def forward(self, x):
        # Convert input tensor to geometric tensor
        x = enn.GeometricTensor(x, self.input_type)
        
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Pool over group and space
        x = x.tensor.mean(dim=[2, 3]) # Global Average Pooling
        
        logits = self.fully_connected(x)
        return logits

def build(config: E2CNNConfig):
    return E2EquivariantCNN(config)
