
import shutil
import mlflow
from pathlib import Path
import logging
from mlflow.models.signature import infer_signature
import numpy as np
import os
from PIL import Image
import torch
from typing import Tuple, Any

#custom imports
from create_logger import get_logger
from MODEL_config_loader import MlflowConfiguration
from models_custom_module import ModelWithPipeline, load_model_and_pipeline


def register_model(
    model_uri: str,
    pipeline_uri: str,
    model_name: str,
    accuracy: float,
    eval_set: str = "base"
):
    """
    Wraps the model with its pipeline, infers the signature using dummy data,
    and registers it to the MLflow Model Registry.
    """
    # CONSTANTS
    MODULE_NAME  = 'MODEL_REGISTER'
    LOGGER_LEVEL = logging.INFO
    logger = get_logger(MODULE_NAME, MODULE_NAME, LOGGER_LEVEL)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        mlflow_config = MlflowConfiguration()
        tracking_uri = mlflow_config.tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("register_model")
    
    try:
        logger.info(f"Loading model from {model_uri} and pipeline from {pipeline_uri}")
        model, pipeline = load_model_and_pipeline(model_uri, pipeline_uri)
        wrapped_model = ModelWithPipeline(model,pipeline)
        wrapped_model.eval()
        
    except Exception as e:
        logger.error(f"An error occurred during loading of parsed model and pipeline {e}")
        raise

    dummy_numpy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_image = Image.fromarray(dummy_numpy)

    with mlflow.start_run() as run:
        with torch.no_grad():
            output_tensor = wrapped_model(dummy_image)
        signature = infer_signature(dummy_numpy, output_tensor.cpu().numpy())

        artifact_path = "model_with_pipeline"
        mlflow.pytorch.log_model(
            wrapped_model,
            artifact_path,
            signature=signature,
            pip_requirements=["torch", "torchvision", "numpy", "Pillow"]
        )

        mlflow.log_params({
            "original_model_uri": model_uri,
            "original_pipeline_uri": pipeline_uri,
            "parsed_accuracy": accuracy,
            "eval_set": eval_set,
        })

        run_id = run.info.run_id
        model_uri_artifact = f"runs:/{run_id}/{artifact_path}"
        result = mlflow.register_model(model_uri_artifact, model_name)
        logger.info(f'Model {model_name} registered with success - version {result.version}')
    
    mlflow_client = mlflow.MlflowClient()
    mlflow_client.set_model_version_tag(model_name, result.version, "accuracy",accuracy )
    mlflow_client.set_model_version_tag(model_name, result.version, "eval_set",eval_set )
    logger.info(f"Assigning alias 'production' to version {result.version} of model {model_name}")
    mlflow_client.set_registered_model_alias(model_name, "production", result.version)