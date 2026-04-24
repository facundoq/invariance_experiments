from comparison_experiment import ModelComparisonExperiment
from experiment_base import RunConfig
from models import resnet, vit, tipooling, e2cnn, rivit, cnn_rotperm

# Model names for scaling (timm/torchvision conventions)
# ViT: tiny, small, base
# ResNet: 10, 18, 34, 50

# MNIST configurations (Smallest)
mnist_configs = [
    RunConfig(
        run_name="resnet_mnist",
        build_fn=resnet.build,
        model_config=resnet.ResNetConfig(n_classes=10, model_name='resnet10t'), # resnet10 is very small
        dataset_name="mnist",
        epochs=1, lr=0.001, batch_size=32
    ),
    RunConfig(
        run_name="rotperm_mnist",
        build_fn=cnn_rotperm.build,
        model_config=cnn_rotperm.RotPermConfig(
            n_classes=10,
            topo={
                'conv1': {'r4': 4,   'r2': 0,   'r0': 0,                },
                'mod1':  {'r4': 4,   'r2': 0,   'r0': 0,   "swap": True },
                'mod2':  {'r4': 8,   'r2': 0,   'r0': 0,   "swap": True },
                'mod3':  {'r4': 16,  'r2': 0,   'r0': 0,   "swap": True },
                'mod4':  {'r4': 32,  'r2': 0,   'r0': 0,   "swap": True }
            }
        ),
        dataset_name="mnist",
        epochs=1, lr=0.001, batch_size=32
    ),
    RunConfig(
        run_name="e2cnn_mnist",
        build_fn=e2cnn.build,
        model_config=e2cnn.E2CNNConfig(n_classes=10, width_scale=2),
        dataset_name="mnist",
        epochs=1, lr=0.001, batch_size=32
    ),
    RunConfig(
        run_name="vit_mnist",
        build_fn=vit.build,
        model_config=vit.ViTConfig(n_classes=10, model_name='vit_tiny_patch16_224'),
        dataset_name="mnist",
        epochs=1, lr=0.0001, batch_size=32
    )
]

# CIFAR10 configurations (Medium-Small)
cifar10_configs = [
    RunConfig(
        run_name="resnet_cifar10",
        build_fn=resnet.build,
        model_config=resnet.ResNetConfig(n_classes=10, model_name='resnet18'),
        dataset_name="cifar10",
        epochs=1, lr=0.001, batch_size=32
    ),
    RunConfig(
        run_name="rotperm_cifar10",
        build_fn=cnn_rotperm.build,
        model_config=cnn_rotperm.RotPermConfig(
            n_classes=10,
            topo={
                'conv1': {'r4': 16,   'r2': 0,   'r0': 0,                },
                'mod1':  {'r4': 16,   'r2': 0,   'r0': 0,   "swap": True },
                'mod2':  {'r4': 32,   'r2': 0,   'r0': 0,   "swap": True },
                'mod3':  {'r4': 64,   'r2': 0,   'r0': 0,   "swap": True },
                'mod4':  {'r4': 128,  'r2': 0,   'r0': 0,   "swap": True }
            }
        ),
        dataset_name="cifar10",
        epochs=1, lr=0.001, batch_size=32
    ),
    RunConfig(
        run_name="tipooling_cifar10",
        build_fn=tipooling.build,
        model_config=tipooling.TIPoolingConfig(n_classes=10, num_rotations=4, base_config=resnet.ResNetConfig(model_name='resnet18')),
        dataset_name="cifar10",
        epochs=1, lr=0.001, batch_size=32
    )
]

# CIFAR100 configurations (Medium-Large)
cifar100_configs = [
    RunConfig(
        run_name="resnet_cifar100",
        build_fn=resnet.build,
        model_config=resnet.ResNetConfig(n_classes=100, model_name='resnet34'),
        dataset_name="cifar100",
        epochs=1, lr=0.001, batch_size=32
    ),
    RunConfig(
        run_name="rotperm_cifar100",
        build_fn=cnn_rotperm.build,
        model_config=cnn_rotperm.RotPermConfig(
            n_classes=100,
            topo={
                'conv1': {'r4': 16,   'r2': 0,   'r0': 0,                },
                'mod1':  {'r4': 16,   'r2': 0,   'r0': 0,   "swap": True },
                'mod2':  {'r4': 32,   'r2': 0,   'r0': 0,   "swap": True },
                'mod3':  {'r4': 64,   'r2': 0,   'r0': 0,   "swap": True },
                'mod4':  {'r4': 128,  'r2': 0,   'r0': 0,   "swap": True }
            }
        ),
        dataset_name="cifar100",
        epochs=1, lr=0.001, batch_size=32
    ),
    RunConfig(
        run_name="rivit_cifar100",
        build_fn=rivit.build,
        model_config=rivit.RI_ViTConfig(n_classes=100, model_name='vit_tiny_patch16_224'),
        dataset_name="cifar100",
        epochs=1, lr=0.0001, batch_size=32
    )
]

# TinyImageNet configurations (Largest)
tinyimagenet_configs = [
    RunConfig(
        run_name="resnet_tinyimagenet",
        build_fn=resnet.build,
        model_config=resnet.ResNetConfig(n_classes=200, model_name='resnet50'),
        dataset_name="tinyimagenet",
        epochs=1, lr=0.001, batch_size=16
    ),
    RunConfig(
        run_name="rotperm_tinyimagenet",
        build_fn=cnn_rotperm.build,
        model_config=cnn_rotperm.RotPermConfig(
            n_classes=200,
            topo={
                'conv1': {'r4': 32,   'r2': 0,   'r0': 0,                },
                'mod1':  {'r4': 32,   'r2': 0,   'r0': 0,   "swap": True },
                'mod2':  {'r4': 64,   'r2': 0,   'r0': 0,   "swap": True },
                'mod3':  {'r4': 128,  'r2': 0,   'r0': 0,   "swap": True },
                'mod4':  {'r4': 256,  'r2': 0,   'r0': 0,   "swap": True }
            }
        ),
        dataset_name="tinyimagenet",
        epochs=1, lr=0.001, batch_size=16
    ),
    RunConfig(
        run_name="vit_tinyimagenet",
        build_fn=vit.build,
        model_config=vit.ViTConfig(n_classes=200, model_name='vit_small_patch16_224'),
        dataset_name="tinyimagenet",
        epochs=1, lr=0.0001, batch_size=16
    )
]

test_configs = mnist_configs + cifar10_configs + cifar100_configs + tinyimagenet_configs

if __name__ == "__main__":
    print("Starting parameter-scaled smoke test across all datasets...")
    experiment = ModelComparisonExperiment("RunConfig_Scaled_MultiDataset_Test", test_configs)
    experiment.run(subset_size=10, test_subset_size=10)
    print("All tests completed successfully!")
