"""
This module contains the functionality for training and evaluate the models defined in models_custom_module.
It uses MLflow for experiment tracking and logging, enabling the analysis of model performance and
parameter tuning over various runs. The module integrates several steps including data preparation, model 
building, training, evaluation, and logging.
"""

import gc
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import os
from pathlib import Path
import pickle
import tempfile
import torch

#custom import
from MODEL_data_loader import DataLoader_BTI
from MODEL_config_loader import MlflowConfiguration, MultistepConfig, SingleStepConfig
from models_custom_module import get_train_model


def apply_block_unfreezing(pl_module: pl.LightningModule, num_blocks_to_unfreeze: int, logger):
    """
    Unfreezes the final 'n' blocks of the feature extractor.
    Adapts to 'unfreeze_blocks': 0 (Head Only), -1 (All), N (Last N blocks).
    """
    feature_extractor = pl_module.model.features
    child_layers = list(feature_extractor.children())
    total_layers = len(child_layers)

    logger.info(f"Unfreezing classifier parameters")
    for param in pl_module.model.classifier.parameters():
        param.requires_grad = True

    logger.info(f"\n--- Unfreeze Configuration: {num_blocks_to_unfreeze} blocks ---")

    # CASE A: Unfreeze EVERYTHING (-1)
    if num_blocks_to_unfreeze == -1:
        logger.info("Unfreezing ALL layers.")
        for param in feature_extractor.parameters():
            param.requires_grad = True
        return

    # freeze everything to be safe that some features aren't intentionally unfrozen
    for param in feature_extractor.parameters():
        param.requires_grad = False

    # CASE B: Frozen Backbone (0) - Only Classifier is trainable
    if num_blocks_to_unfreeze == 0:
        logger.info("Freezing ALL backbone layers (Training Head Only).")
        return

    # CASE C: Partial Unfreeze
    if num_blocks_to_unfreeze > total_layers:
        logger.info(f"Warning: Requested {num_blocks_to_unfreeze} blocks, but model only has {total_layers}. Unfreezing all.")
        num_blocks_to_unfreeze = total_layers

    # Select the last N layers
    layers_to_train = child_layers[-num_blocks_to_unfreeze:]
    logger.info(f"-> Unfreezing last {num_blocks_to_unfreeze} blocks.")

    for layer in layers_to_train:
        for param in layer.parameters():
            param.requires_grad = True


def log_sample_images(data_module_sample: DataLoader_BTI):
    data_module_sample.setup()
    train_data_sample =data_module_sample.train_dataloader() 
    images, _ = next(iter(train_data_sample))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Take max 8 images
    batch_to_show = images[:8]
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    axs = axs.flatten()

    for i, img in enumerate(batch_to_show):
        img = img * std + mean  # Unnormalize
        np_img = img.permute(1, 2, 0).numpy()
        # Clip to valid range [0,1]
        np_img = np_img.clip(0, 1)

        axs[i].imshow(np_img)
        axs[i].axis('off')

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        plt.tight_layout()
        plt.savefig(tmp.name)
        mlflow.log_artifact(tmp.name, "image_samples")
        tmp_path = Path(tmp.name)

    plt.close(fig)
    Path(tmp_path).unlink()


#Train with mlflow and saving in a sqlite DB
#run on cmd: mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host <HOST ADRESS> --port <PORT>                 
def train_evaluate_mlflow(logger):
    """
    Train and evaluate a machine learning model using MLflow, and log the process in an MLflow server.

    The parameters are loaded from config_loader.py
    """
    
    mlflow_config = MlflowConfiguration()

    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            tracking_uri = mlflow_config.tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(mlflow_config.exp_name)
        
    except Exception as e:
        logger.error(f"Problem loading mlflow experiment and tracking uri: {e}")
        raise

    multi_config = MultistepConfig()
    single_config = SingleStepConfig()
    # If multistep is disabled, create a dummy single step using global config
    steps = multi_config.steps if multi_config.enable_multistep else single_config.steps
    model_name = steps[0].model_name

    try:
        model = get_train_model(model_name=model_name, logger=logger, model_params={'learning_rate': steps[0].lr})

    except Exception as e:
        logger.error(f"Unexpected error during model building: {e}")
        raise

    data_loader = DataLoader_BTI()
    data_loader.setup()

    pipeline_path = Path("transf_pipeline.pkl")
    with open(pipeline_path, "wb") as file:
        pickle.dump(data_loader.val_transform, file)

    model_uri = None
    transf_pipeline_uri = None
    checkpoint_dirpath_base = Path('runs_checkpoints') / mlflow_config.run_name
    # Possible improvements:
    #   - Evaluate each step and return the best performance step (currently return last)
    #   - Autostop if decrease accuracy in one of the steps
    for step in steps:
        logger.info(f"Step {step}")
        run_name = step.step_name+mlflow_config.run_name
        checkpoint_dirpath = checkpoint_dirpath_base.joinpath(step.step_name)
        checkpoint_dirpath.mkdir(parents=True, exist_ok=True)
        base_filename = f"{model_name}-{step.step_name}-"
        model_checkpoint = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename=base_filename+"-{epoch:03d}-{val_loss:.2f}",
            auto_insert_metric_name=False,
            save_weights_only=True,
            save_top_k=1,
            dirpath=checkpoint_dirpath,
            save_last=True,
        )
        model_callbacks = [
            EarlyStopping(monitor="val_loss", patience=30),
            model_checkpoint,
        ]
        trainer = pl.Trainer(
            accelerator="auto",
            devices="auto",
            strategy="auto",
            max_epochs=step.epochs,
            precision=step.precision,
            callbacks=model_callbacks,
        )

        apply_block_unfreezing(model, step.unfreeze_blocks, logger)
        with mlflow.start_run(run_name= run_name):
            mlflow.pytorch.autolog(log_models=False)
            mlflow.log_params({
                "model_name": step.model_name,
                "epochs": step.epochs,
                "dataset": data_loader.config.dataset_name,
                "train_transform": data_loader.train_transform,
            })
            log_sample_images(data_loader)

            mlflow.log_artifact(pipeline_path)

            trainer.fit(model, data_loader)
            #mlflow.log_artifact('lightning_logs')
            mlflow.set_tags(mlflow_config.tags)

            #load the best checkpoints weights to the model
            best_checkpoint_path = model_checkpoint.best_model_path
            if best_checkpoint_path:
                logger.info(f"Loading best model from: {best_checkpoint_path}")
                model = get_train_model(
                    model_name=model_name,
                    logger=logger,
                    model_params={'learning_rate': step.lr},
                    checkpoint=best_checkpoint_path
                )
            else:
                logger.warning("No checkpoint found, using last model state.")

            model.eval()
            mlflow.pytorch.log_model(model,"model")

            # Overwrittes, returns the last step
            transf_pipeline_uri = mlflow.get_artifact_uri("transf_pipeline.pkl")
            model_uri = mlflow.get_artifact_uri("model")

    if pipeline_path.exists():
        pipeline_path.unlink()
    model_name = steps[-1].model_name
    return model_uri, transf_pipeline_uri, model_name