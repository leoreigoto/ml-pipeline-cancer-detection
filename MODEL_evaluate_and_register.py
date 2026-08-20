
import argparse
import json
import logging
import mlflow
import numpy as np
import os
from pathlib import Path
import torch
import sys

#custom imports
from create_logger import get_logger
from MODEL_config_loader import MlflowConfiguration
from models_custom_module import load_model_and_pipeline
from MODEL_data_loader import DataLoader_BTI
from MODEL_register_model import register_model


def evaluate_model(model, pipeline, logger):
    eval_dataloader = DataLoader_BTI()
    eval_dataloader.setup(stage="test")
    eval_dataloader.test_dataset.transform = pipeline
    
    eval_data = eval_dataloader.test_dataloader()
    all_predictions = []
    all_labels = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    logger.info(f"Starting evaluation on {device}...")

    # Disable gradient calculation for inference
    with torch.no_grad():
        for inputs, labels in eval_data:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Perform inference
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float().squeeze()
            
            # Collect predictions and labels
            if preds.ndim == 0:
                preds = preds.unsqueeze(0)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = (np.array(all_predictions) == np.array(all_labels)).mean()
    return accuracy

# Won't be compatible with the new microservices architecture because it uses DockerOperator (it requires to parse
# the xcom as args instead)
# I left it to keep backwards compatibility with the previous DAG (used in the article)
def get_args_from_xcom(**kwargs):
    """Retrieves arguments from Airflow XComs."""
    ti = kwargs.get('ti')
    if not ti:
        raise ValueError("Airflow TaskInstance (ti) not found in kwargs")

    model_uri = ti.xcom_pull(task_ids='model_train_task', key='model_uri')
    pipeline_uri = ti.xcom_pull(task_ids='model_train_task', key='transf_pipeline_uri')
    model_name = ti.xcom_pull(task_ids='model_train_task', key='model_name')

    return model_uri, pipeline_uri, model_name

def main(model_uri, pipeline_uri, model_name, **kwargs):
    MODULE_NAME  = 'MODEL_EVALUATOR'
    LOGGER_LEVEL = logging.INFO
    logger = get_logger(MODULE_NAME, MODULE_NAME, LOGGER_LEVEL)
    prod_model, prod_model_name, prod_model_version = None, None, None
    is_promoted = False

    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            mlflow_config = MlflowConfiguration()
            tracking_uri = mlflow_config.tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow_client = mlflow.MlflowClient()
        registered_models = mlflow_client.search_registered_models()
        for model in registered_models:
            if "production" in model.aliases:
                prod_model_name = model.name
                prod_model_version = model.aliases['production']
                logger.info(f"Found Production model: {prod_model_name} (Version: {prod_model_version})")
                prod_model_uri = f"models:/{prod_model_name}/{prod_model_version}"
                device = "cuda" if torch.cuda.is_available() else "cpu"
                prod_model = mlflow.pytorch.load_model(prod_model_uri,map_location=torch.device(device))
                prod_model.eval()
                break

        if prod_model is None:
            logger.warning("No model found with alias 'production'. The new model will be promoted automatically.")

    except Exception as e:
        logger.error(f"An error occurred during loading of production model {e}")
        raise
    
    # Load both the recent model and the production model with their pipelines
    try:
        curr_model, curr_pipeline = load_model_and_pipeline(model_uri, pipeline_uri)
        curr_model.eval()
        
    except Exception as e:
        logger.error(f"An error occurred during loading of parsed model and pipeline {e}")
        raise

    try:
        logger.info("Evaluating candidate model...")
        recent_accuracy = evaluate_model(curr_model, curr_pipeline, logger)
        logger.info(f"Candidate Model Accuracy: {recent_accuracy:.4f}")

        prod_accuracy = 0.0
        if prod_model:
            logger.info("Evaluating production model...")
            # prod_model is ModelWithPipeline
            prod_accuracy = evaluate_model(prod_model.model, prod_model.pipeline, logger)
            logger.info(f"Production Model Accuracy: {prod_accuracy:.4f}")
            mlflow_client.set_model_version_tag(prod_model_name, prod_model_version, "last_eval_accuracy", prod_accuracy)

    except Exception as e:
        logger.error(f"An error occurred during model evaluation {e}")
        raise

    if recent_accuracy > prod_accuracy:
        if prod_model is None:
            logger.info("No production model exists. Promoting candidate model.")
        else:
            logger.info(f"Candidate ({recent_accuracy:.4f}) > Production ({prod_accuracy:.4f}). Promoting candidate model.")
            mlflow_client.delete_registered_model_alias(prod_model_name, "production")

        register_model(
            model_uri=model_uri,
            pipeline_uri=pipeline_uri,
            model_name=model_name,
            accuracy=recent_accuracy,
            eval_set="base"
            )
        is_promoted = True
    else:
        logger.info("The production model performs better than the recently trained model.")

    # For Airflow Dag xcom
    output_data = {
        "promoted": is_promoted,
    }
    return output_data

        
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_uri", type=str, required=False, help="MLFlow URI of the model")
    parser.add_argument("--pipeline_uri", type=str, required=False, help="MLFlow URI of the transformation pipeline")
    parser.add_argument("--model_name", type=str, required=False, help="Name of the model to be evaluated")
    
    args = parser.parse_args()
    
    if args.model_uri is None or args.pipeline_uri is None:
        model_uri, pipeline_uri, model_name = get_args_from_xcom()
    else:
        if args.model_name is None:
            args.model_name = 'Unidentified model'
        model_uri, pipeline_uri, model_name = args.model_uri, args.pipeline_uri, args.model_name

    output_data = main(model_uri, pipeline_uri, model_name)

    # The DockerOperator captures the last line printed to stdout as the XCom value
    logging.shutdown()
    sys.stdout.flush()
    print(json.dumps(output_data))
    sys.stdout.flush()

