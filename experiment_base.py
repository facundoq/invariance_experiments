import os
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from typing import Any, Callable

@dataclass
class DeviceConfig:
    """Configuration for the computing device (CPU, CUDA, ROCm)."""
    device_type: str = "auto"  # "cpu", "cuda", "rocm", "auto"
    use_rocm: bool = True     # Explicitly use ROCm for RX 6600
    gfx_version: str = "10.3.0" # Required for RX 6600

    def setup_environment(self):
        if self.use_rocm:
             print(f"Setting ROCm override: HSA_OVERRIDE_GFX_VERSION={self.gfx_version}")
             os.environ["HSA_OVERRIDE_GFX_VERSION"] = self.gfx_version

    def get_torch_device(self):
        if self.device_type != "auto":
            return torch.device(self.device_type)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class RunConfig:
    """Configuration for a single experiment run."""
    run_name: str
    build_fn: Callable[[Any], nn.Module]
    model_config: Any
    dataset_name: str = "cifar10"
    epochs: int = 2
    lr: float = 0.001
    batch_size: int = 32

class Experiment:
    """
    Base class for MLflow experiments.
    Encapsulates MLflow operations.
    """
    def __init__(self, experiment_name: str, device_config: DeviceConfig = None):
        self.experiment_name = experiment_name
        self.device_config = device_config or DeviceConfig()
        self.device_config.setup_environment()
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str):
        return mlflow.start_run(run_name=run_name)

    def log_config(self, config):
        """Logs a dataclass config as parameters."""
        if hasattr(config, "__dataclass_fields__"):
            mlflow.log_params(asdict(config))
        elif isinstance(config, dict):
            mlflow.log_params(config)

    def log_metrics(self, metrics: dict, step: int = None):
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: nn.Module, artifact_path: str = "model"):
        mlflow.pytorch.log_model(model, artifact_path)

    def get_device(self):
        return self.device_config.get_torch_device()
