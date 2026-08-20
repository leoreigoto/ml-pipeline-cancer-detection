"""
Configuration Loader Module .

This module centralizes configuration settings for Data, Training, and MLflow
experiments. It utilizes Python dataclasses for immutability and type safety,
and provides a transformation pipeline generator.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from torchvision.transforms import v2
from typing import Dict, List

SHARED_MODEL_NAME = "EffnetB0"

def _get_timestamp() -> str:
    """Returns the current timestamp in 'YYYY-MM-DD_HH-MM-SS' format."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

_CURRENT_RUN_ID = f"RUN-FAKE-TRAINING-DATA-2IMAGES_{_get_timestamp()}"


@dataclass(frozen=True)
class DataConfiguration:
    """Configuration for data paths and loader settings."""
    _base_root =Path("DATA") / "breast_thermal_frontal"  # Path("DATA") / "breast_thermal_frontal"  or Path("DATA") / "breast_thermal_segmented"

    train_data_root: Path = _base_root / "Train"
    val_data_root: Path = _base_root / "Test"
    eval_data_root: Path = _base_root / "Test" # for segmented: _base_root / "Evaluate"
    num_workers: int = 8
    batch_size: int = 16
    num_classes: int = 2
    dataset_name: str = "fake_training_data_2images"


@dataclass(frozen=True)
class StepConfig:
    """Configuration for a single training step."""
    step_name: str = "default_step_config"
    model_name: str = SHARED_MODEL_NAME
    epochs: int  = 1 # 220
    lr: float = 1e-4
    unfreeze_blocks: int  = 0  # 0 = Head only, -1 = All, 2 = Last 2 blocks
    precision: str = "16-mixed"


@dataclass(frozen=True)
class SingleStepConfig:
    """Configuration for model hyperparameters and training settings."""
    epochs: int = 1  # 220
    lr: float = 1e-4
    unfreeze_blocks: int = 0
    steps: List[StepConfig] = field(default_factory=lambda: [
        StepConfig(
            step_name="unique_step_",
            model_name=SHARED_MODEL_NAME,  # Hardcoded default matching above
            epochs=1,
            lr=1e-4,
            unfreeze_blocks=0
        ),
    ])



@dataclass(frozen=True)
class MultistepConfig_sanity_check:
    """Master configuration for the multi-stage training process."""
    enable_multistep: bool = True # if False does a single step training using TrainingConfiguration config
    steps: List[StepConfig] = field(default_factory=lambda: [
        # Step 1: Train Head Only (High LR, fewer epochs)
        StepConfig(step_name="s1_head_", model_name = SHARED_MODEL_NAME, epochs=1, lr=1e-3, unfreeze_blocks=0),
        # Step 2: Unfreeze Last 2 Blocks and lower LR
        StepConfig(step_name="s2_last2_", model_name = SHARED_MODEL_NAME, epochs=1, lr=1e-4, unfreeze_blocks=2),
        # Step 3: Unfreeze Last 3 Blocks
        StepConfig(step_name="s3_last3_", model_name = SHARED_MODEL_NAME, epochs=1, lr=1e-4, unfreeze_blocks=3),
        # Step 4: Unfreeze Last 4 Blocks
        StepConfig(step_name="s4_last4_", model_name = SHARED_MODEL_NAME, epochs=1, lr=1e-4, unfreeze_blocks=4),
        # Step 5: Unfreeze Last 5 Blocks
        StepConfig(step_name="s5_last5_", model_name = SHARED_MODEL_NAME, epochs=1, lr=1e-4, unfreeze_blocks=5),
        # Step 6: Lower LR
        StepConfig(step_name="s6_last5_lowerlr_", model_name = SHARED_MODEL_NAME, epochs=1, lr=1e-5, unfreeze_blocks=5),
    ])


@dataclass(frozen=True)
class MultistepConfig:
    """Master configuration for the multi-stage training process."""
    enable_multistep: bool = True # if False does a single step training using TrainingConfiguration config
    steps: List[StepConfig] = field(default_factory=lambda: [
        # Step 1: Train Head Only (High LR, fewer epochs)
        StepConfig(step_name="s1_head_", model_name = SHARED_MODEL_NAME, epochs=30, lr=1e-3, unfreeze_blocks=0),
        # Step 2: Unfreeze Last 2 Blocks and lower LR
        StepConfig(step_name="s2_last2_", model_name = SHARED_MODEL_NAME, epochs=30, lr=1e-4, unfreeze_blocks=2),
        # Step 3: Unfreeze Last 3 Blocks
        StepConfig(step_name="s3_last3_", model_name = SHARED_MODEL_NAME, epochs=30, lr=1e-4, unfreeze_blocks=3),
        # Step 4: Unfreeze Last 4 Blocks
        StepConfig(step_name="s4_last4_", model_name = SHARED_MODEL_NAME, epochs=30, lr=1e-4, unfreeze_blocks=4),
        # Step 5: Unfreeze Last 5 Blocks
        StepConfig(step_name="s5_last5_", model_name = SHARED_MODEL_NAME, epochs=30, lr=1e-4, unfreeze_blocks=5),
        # Step 6: Lower LR
        StepConfig(step_name="s6_last5_lowerlr_", model_name = SHARED_MODEL_NAME, epochs=40, lr=1e-5, unfreeze_blocks=5),
    ])


@dataclass(frozen=True)
class MlflowConfiguration:
    """
    Configuration for MLflow experiment tracking.

    To start the server:
    mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1 --port 5000
    """
    # Overwritten with MLFLOW_TRACKING_URI (configured in the docker container)
    tracking_uri: str = 'http://127.0.0.1:5000'
    artifact_root: Path = Path("./artifacts")
    exp_name: str = "Default_run"
    tags: Dict[str, str] = field(default_factory=lambda: {
        "tag1": "Multistep_Training",
        "tag2": "Effnetb0"
    })
    run_name: str = _CURRENT_RUN_ID


def get_transforms_pipeline(resize_dim: int = 300, crop_dim: int = 480, aug: bool =False) -> v2.Compose:
    """Returns the training/inference pipeline."""
    # initial transforms
    transforms_list = [
        v2.Resize(resize_dim),
        v2.Pad(padding=(40, 90, 40, 90), fill=0, padding_mode='constant'),
        v2.CenterCrop(crop_dim),
    ]

    # aug transforms
    if aug:
        transforms_list.extend([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=30),
            v2.RandomAffine(degrees=0, scale=(0.85, 1.15)),
        ])

    # final transforms
    transforms_list.extend([
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return v2.Compose(transforms_list)