import torch.nn as nn
import timm
from dataclasses import dataclass

@dataclass
class ViTConfig:
    n_classes: int = 10
    model_name: str = 'vit_tiny_patch16_224'
    pretrained: bool = False

def build(config: ViTConfig):
    return timm.create_model(config.model_name, pretrained=config.pretrained, num_classes=config.n_classes)
