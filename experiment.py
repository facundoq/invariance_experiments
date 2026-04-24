from comparison_experiment import ModelComparisonExperiment
from experiment_base import RunConfig
from models import resnet, vit, tipooling, e2cnn, rivit, cnn_rotperm

# Define the experiment matrix as a list of RunConfig instances
configs = [
    RunConfig(
        run_name="resnet_baseline",
        build_fn=resnet.build,
        model_config=resnet.ResNetConfig(n_classes=10),
        epochs=2,
        lr=0.001,
        batch_size=32
    ),
    RunConfig(
        run_name="cnn_rotperm_4rot",
        build_fn=cnn_rotperm.build,
        model_config=cnn_rotperm.RotPermConfig(n_classes=10),
        epochs=2,
        lr=0.001,
        batch_size=32
    ),
    RunConfig(
        run_name="tipooling_8rot",
        build_fn=tipooling.build,
        model_config=tipooling.TIPoolingConfig(num_rotations=8),
        epochs=2,
        lr=0.001,
        batch_size=16
    ),
    RunConfig(
        run_name="e2cnn_c8_equivariant",
        build_fn=e2cnn.build,
        model_config=e2cnn.E2CNNConfig(num_rotations=8, input_size=224),
        epochs=2,
        lr=0.001,
        batch_size=32
    ),
    RunConfig(
        run_name="vit_baseline",
        build_fn=vit.build,
        model_config=vit.ViTConfig(n_classes=10),
        epochs=2,
        lr=0.0001,
        batch_size=32
    ),
    RunConfig(
        run_name="rivit_4rot",
        build_fn=rivit.build,
        model_config=rivit.RI_ViTConfig(num_rotations=4),
        epochs=2,
        lr=0.0001,
        batch_size=16
    )
]

if __name__ == "__main__":
    experiment = ModelComparisonExperiment("Invariance_Comparison_Modular", configs)
    experiment.run(subset_size=500, test_subset_size=100)
