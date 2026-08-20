
from datetime import datetime
import logging
import mlflow
import os
from pathlib import Path
import pickle

# custom imports
from create_logger import get_logger
from models_custom_module import get_train_model
from MODEL_data_loader import DataLoader_BTI
from MODEL_register_model import register_model


def main():
    MODULE_NAME = 'export_from_local_weights'
    LOGGER_LEVEL = logging.INFO
    CHECKPOINT_PATH = Path('DATA') / 'temporary_checkpoint' / 'frontal_latest_checkpoint.pth'
    logger = get_logger(MODULE_NAME, MODULE_NAME, LOGGER_LEVEL)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)
    run_name = f"export_from_local_weights_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    mlflow.set_experiment(run_name)
    mlflow_client = mlflow.MlflowClient()
    checkpoint_str = str(CHECKPOINT_PATH)
    model = get_train_model(
        model_name="EffnetB0",
        logger=logger,
        checkpoint=checkpoint_str
    )
    data_loader = DataLoader_BTI()
    data_loader.setup()
    pipeline_path = Path("transf_pipeline.pkl")
    with open(pipeline_path, "wb") as file:
        pickle.dump(data_loader.val_transform, file)

    model_uri = None
    transf_pipeline_uri = None
    with mlflow.start_run(run_name=run_name):
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifact(pipeline_path)
        transf_pipeline_uri = mlflow.get_artifact_uri("transf_pipeline.pkl")
        model_uri = mlflow.get_artifact_uri("model")


    if pipeline_path.exists():
        pipeline_path.unlink()

    register_model(
        model_uri=model_uri,
        pipeline_uri=transf_pipeline_uri,
        model_name="model_force_save_local_weight",
        accuracy="skipped",
        eval_set="skipped"
    )
if __name__ == "__main__":
    main()
