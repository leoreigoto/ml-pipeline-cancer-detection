from huggingface_hub import HfApi
import logging
import mlflow.pytorch
import os
import torch

from create_logger import get_logger

MODULE_NAME  = 'UFF_TRAIN_SERVER'
LOGGER_LEVEL = logging.INFO
logger = get_logger(MODULE_NAME, MODULE_NAME, LOGGER_LEVEL)

OUTPUT_PATH = "/mlflow_data/model_final.pth"

HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "leoreigoto/termografia_demo"


def export_and_upload():
    prod_model = None
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
                prod_model = mlflow.pytorch.load_model(prod_model_uri,map_location='cpu')
                prod_model.eval()
                break

    except Exception as e:
        logger.error(f"An error occurred during loading of production model {e}")
        raise

    if prod_model is None:
        logger.error("No production model found")
        raise Exception("No production model found")

    #torch.save(prod_model.state_dict(), OUTPUT_PATH)
    torch.save(prod_model, OUTPUT_PATH)

    if HF_TOKEN:
        print(f"Initiating upload to Hugging Face: {REPO_ID}...")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=OUTPUT_PATH,
            path_in_repo="model_final.pth",
            repo_id=REPO_ID,
            repo_type="space",
            token=HF_TOKEN
        )
        print("Uploaded concluded with success!")
    else:
        print("HF_TOKEN not found. Error to upload to Hugging Face.")


if __name__ == "__main__":
    export_and_upload()