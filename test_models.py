from comparison_experiment import ModelComparisonExperiment
from experiment_base import RunConfig
from models import resnet, vit, tipooling, e2cnn, rivit, cnn_rotperm

# Minimal test configuration for smoke testing using RunConfig instances
test_configs = [
    # CIFAR10 tests
    RunConfig(
        run_name="test_resnet_cifar10",
        build_fn=resnet.build,
        model_config=resnet.ResNetConfig(n_classes=10),
        dataset_name="cifar10",
        epochs=1,
        lr=0.001,
        batch_size=32
    ),
    RunConfig(
        run_name="test_rotperm_cifar10",
        build_fn=cnn_rotperm.build,
        model_config=cnn_rotperm.RotPermConfig(n_classes=10),
        dataset_name="cifar10",
        epochs=1,
        lr=0.001,
        batch_size=32
    ),
    # MNIST tests (Small topology)
    RunConfig(
        run_name="test_rotperm_mnist",
        build_fn=cnn_rotperm.build,
        model_config=cnn_rotperm.RotPermConfig(
            n_classes=10,
            topo={
                'conv1': {'r4': 8,   'r2': 0,   'r0': 0,                },
                'mod1':  {'r4': 8,   'r2': 0,   'r0': 0,   "swap": True },
                'mod2':  {'r4': 16,  'r2': 0,   'r0': 0,   "swap": True },
                'mod3':  {'r4': 32,  'r2': 0,   'r0': 0,   "swap": True },
                'mod4':  {'r4': 64,  'r2': 0,   'r0': 0,   "swap": True }
            }
        ),
        dataset_name="mnist",
        epochs=1,
        lr=0.001,
        batch_size=32
    ),
    # CIFAR100 tests (Standard topology)
    RunConfig(
        run_name="test_rotperm_cifar100",
        build_fn=cnn_rotperm.build,
        model_config=cnn_rotperm.RotPermConfig(n_classes=100),
        dataset_name="cifar100",
        epochs=1,
        lr=0.001,
        batch_size=32
    ),
    # TinyImageNet tests (Large topology)
    RunConfig(
        run_name="test_rotperm_tinyimagenet",
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
        epochs=1,
        lr=0.001,
        batch_size=16
    ),
    # Other models test
    RunConfig(
        run_name="test_tipooling",
        build_fn=tipooling.build,
        model_config=tipooling.TIPoolingConfig(num_rotations=2),
        epochs=1,
        lr=0.001,
        batch_size=32
    ),
    RunConfig(
        run_name="test_e2cnn",
        build_fn=e2cnn.build,
        model_config=e2cnn.E2CNNConfig(num_rotations=4, input_size=224),
        epochs=1,
        lr=0.001,
        batch_size=32
    ),
    RunConfig(
        run_name="test_vit",
        build_fn=vit.build,
        model_config=vit.ViTConfig(n_classes=10),
        epochs=1,
        lr=0.0001,
        batch_size=32
    ),
    RunConfig(
        run_name="test_rivit",
        build_fn=rivit.build,
        model_config=rivit.RI_ViTConfig(num_rotations=2),
        epochs=1,
        lr=0.0001,
        batch_size=32
    )
]

if __name__ == "__main__":
    print("Starting smoke test for RunConfig based architecture with multiple datasets...")
    experiment = ModelComparisonExperiment("RunConfig_MultiDataset_Test", test_configs)
    # Using very small subsets for quick smoke testing
    experiment.run(subset_size=10, test_subset_size=10)
    print("Smoke test completed successfully!")
