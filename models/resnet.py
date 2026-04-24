import torch.nn as nn
import timm
from dataclasses import dataclass

@dataclass
class ResNetConfig:
    n_classes: int = 10
    model_name: str = 'resnet18'
    pretrained: bool = False

def build(config: ResNetConfig):
    return timm.create_model(config.model_name, pretrained=config.pretrained, num_classes=config.n_classes)
