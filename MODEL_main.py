
import json
import lightning.pytorch as pl
import logging
import numpy as np
import random
import torch
import sys

#custom imports
from create_logger import get_logger
from MODEL_mlflow_train import train_evaluate_mlflow


#CONSTANTS
MODULE_NAME  = 'UFF_TRAIN_SERVER'
LOGGER_LEVEL = logging.INFO
SEED = 42

#Train with mlflow and saving in a sqlite DB
#need to start mlflow server
#defaults  MLFLOW_TRACKING_URI= 'http://127.0.0.1:5000'
#cmd: mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host  <HOST ADRESS> --port <PORT> | default: --host  127.0.0.1 --port 5000
# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host  127.0.0.1 --port 5000

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision('high')
    
def main():
    """
    Main function to execute the machine learning pipeline.

    This function initializes logging, loads configuration, 
    reads data, and executes the MLFlow pipeline for training,
    evaluating, and saving the machine learning model.
    """
    logger = get_logger(MODULE_NAME, MODULE_NAME, LOGGER_LEVEL)
    try:
        set_seed(SEED)
    
         # Execute MLFlow pipeline: trains, evaluate and save the model
        model_uri, transf_pipeline_uri, model_name = train_evaluate_mlflow(logger)
        logger.info(f"Pipeline finished successfully.")
        logger.info(f"Model URI: {model_uri}")
        logger.info(f"Pipeline URI: {transf_pipeline_uri}")
        output_data = {
            "model_uri": model_uri,
            "pipeline_uri": transf_pipeline_uri,
            "model_name": model_name
        }


        # The DockerOperator captures the last line printed to stdout as the XCom value
        logging.shutdown()
        sys.stdout.flush()
        print(json.dumps(output_data))
        sys.stdout.flush()
        
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}")
        raise

if __name__ == "__main__":
    main()