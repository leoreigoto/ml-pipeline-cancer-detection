"""
This module integrates FastAPI with MLflow to serve a machine learning model for breast cancer
diagnostic. The API provides endpoints for retrieving model information, performing predictions on image data,
and checking server health. The module incorporates background tasks for updating the model and monitoring 
server status. The model is updated automatically when a new version is assigned the 'production' alias in
the MLflow registry.

Key Components:
- FastAPI application setup with route handlers for '/info', '/predict', and '/health' endpoints.
- Asynchronous tasks for checking model updates and server health.
- Configuration management and logging setup.
- Security measures through API key validation.
"""
import asyncio
import base64
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
import io
import mlflow
import numpy as np
import os
from PIL import Image
from pydantic import BaseModel
import threading
import torch
from typing import AsyncIterator

#custom imports
from API_loggers import get_api_loggers
from API_config_loader import APIConfig, StandardResponse
from API_security_key import validate_api_key


class ModelManager:
    """
    Manages the lifecycle of the ML model to ensure thread safety
    and clean updates.
    """

    def __init__(self):
        self.model = None
        self.name = None
        self.version = None
        self.lock = threading.Lock()

    def load_from_mlflow(self):
        """
        Loads the model from MLflow.
        """
        try:
            # This prioritizes the Docker internal URI, but defaults to API_config.tracking_uri if running outside Docker
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", API_config.tracking_uri)
            mlflow.set_tracking_uri(tracking_uri)
            mlflow_client = mlflow.MlflowClient()

            # Find production model
            registered_models = mlflow_client.search_registered_models()
            target_name = None
            target_version = None

            for model in registered_models:
                if "production" in model.aliases:
                    target_name = model.name
                    target_version = model.aliases['production']
                    break

            if not target_name:
                generic_logger.warning("No model with alias 'production' found in MLflow.")
                return False

            if target_version != self.version or target_name != self.name:
                generic_logger.info(f"Downloading model: {target_name} (Version: {target_version})...")
                uri = f"models:/{target_name}/{target_version}"

                # Heavy blocking I/O
                device = "cuda" if torch.cuda.is_available() else "cpu"
                new_model = mlflow.pytorch.load_model(uri,map_location=torch.device(device))
                new_model.eval()

                with self.lock:
                    self.model = new_model
                    self.name = target_name
                    self.version = target_version

                generic_logger.info(f"Model updated to {target_name} v{target_version}")
                return True

            return False

        except Exception as e:
            generic_logger.error(f"Error loading model from MLflow: {e}")
            # for background updates, we just log and continue.
            return False


# Initialize the manager instance
model_manager = ModelManager()

MODULE_NAME = 'API_UFF'
API_config = APIConfig()

generic_logger, pred_logger  =get_api_loggers(MODULE_NAME)


async def check_model_update():
    """
    Periodically checks for model updates.
    Runs the blocking load function in a separate thread.
    """
    while True:
        try:
            await asyncio.sleep(API_config.model_update_timer)
            generic_logger.info("Checking for model updates...")

            # asyncio.to_thread runs the sync function in a separate thread
            # so the main event loop (handling other requests) doesn't freeze.
            await asyncio.to_thread(model_manager.load_from_mlflow)

        except asyncio.CancelledError:
            generic_logger.info("Update checker task cancelled.")
            break
        except Exception as e:
            generic_logger.error(f"Unexpected error in update loop: {e}")

async def check_server_status():
    """
    Periodically logs server health.
    """
    while True:
        try:
            await asyncio.sleep(API_config.health_check_timer)
            generic_logger.info("Health check: System healthy.")
        except asyncio.CancelledError:
            break
        except Exception as e:
            generic_logger.error(f"Error in health check loop: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manages application startup and shutdown events.
    """
    # Startup: Load model immediately (blocking is okay here, we want to wait before serving)
    generic_logger.info("Startup: Initializing model...")
    # We run this directly to ensure model is ready before app starts accepting requests
    await asyncio.to_thread(model_manager.load_from_mlflow)

    if model_manager.model is None:
        generic_logger.warning("Startup: Application starting WITHOUT a loaded model.")
        #generic_logger.warning("Startup ERROR: Application tried to start WITHOUT a loaded model.")
        #raise RuntimeError("Application failed to start: Model could not be loaded.")

    # Start Background Tasks
    update_task = asyncio.create_task(check_model_update())
    health_task = asyncio.create_task(check_server_status())

    yield  # Application runs and serves requests here

    # Shutdown: Cancel tasks
    generic_logger.info("Shutdown: Stopping background tasks...")
    update_task.cancel()
    health_task.cancel()
    try:
        await update_task
        await health_task
    except asyncio.CancelledError:
        pass

app = FastAPI(
    title="UFF - Breast Cancer Predict - APP",
    lifespan=lifespan
)


@app.get('/info', summary="Get Model Information", response_model=StandardResponse, dependencies=[Depends(validate_api_key)])
async def info(): 
    """
    Returns current model name and version..
    """

    response_info = {
        "name": model_manager.name,
        "version": model_manager.version
    }
    
    return StandardResponse(success=True, endpoint="info", data=response_info)


@app.post('/predict', response_model=StandardResponse, summary="Predict Breast Cancer Diagnosis",
          dependencies=[Depends(validate_api_key)])
def predict(json_payload: dict):
    """
    Runs the model prediction.
    """
    # lock briefly to grab the model reference.
    with model_manager.lock:
        if model_manager.model is None:
            raise HTTPException(status_code=503, detail="Service unavailable: No model loaded.")

        # Capture the model reference locally.
        # Even if background task updates self.model later, 'active_model' points to the old valid object.
        active_model = model_manager.model
        # active_version = model_manager.version

    try:
        # 2. Load Image data
        image_data = base64.b64decode(json_payload['inputs'])
        image = Image.open(io.BytesIO(image_data))

        # 3. Inference
        # The pre-processing transforms are expected to be inside the prod_model() pipeline
        with torch.no_grad():
            output = active_model.forward(image)

        predictions = torch.sigmoid(output).tolist()[0]

        # 4. Logging (Optional)
        # if enable_pred_data_log:
        #        pred_logger.info(f"Prediction: {predictions} (Model v{active_version})")

        return StandardResponse(success=True, endpoint="predict", data={'prediction': predictions})

    except Exception as e:
        generic_logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post('/predict_with_gradcam', response_model=StandardResponse, summary="Predict Breast Cancer Diagnosis",
          dependencies=[Depends(validate_api_key)])
def predict_with_gradcam(json_payload: dict):
    """
    Runs the model prediction with gradcam outít
    """
    # lock briefly to grab the model reference.
    with model_manager.lock:
        if model_manager.model is None:
            raise HTTPException(status_code=503, detail="Service unavailable: No model loaded.")

        # Capture the model reference locally.
        # Even if background task updates self.model later, 'active_model' points to the old valid object.
        active_model = model_manager.model
        # active_version = model_manager.version

    try:
        # Load Image data
        image_data = base64.b64decode(json_payload['inputs'])
        image = Image.open(io.BytesIO(image_data))
        # Inference
        # The pre-processing transforms are expected to be inside the prod_model() pipeline
        output, gradcam_heatmap = active_model.forward_with_gradcam_heatmap(image)

        # Robustly handle scalar or list outputs
        # squeeze() converts [[0.9]] or [0.9] -> 0.9 (scalar)
        predictions = torch.sigmoid(output).detach().cpu().squeeze().tolist()
        # Logging (Optional)
        # if enable_pred_data_log:
        #    pred_logger.info(f"Prediction: {predictions} (Model v{active_version})")

        # Process Heatmap for JSON Response
        #  Remove batch dim [1, 1, H, W] -> [H, W]
        heatmap_np = gradcam_heatmap.detach().cpu().numpy().squeeze()
        # Ensure range 0-255 and uint8 type
        heatmap_np = np.clip(heatmap_np, 0, 1)
        heatmap_uint8 = (heatmap_np * 255).astype(np.uint8)
        # Convert to PIL Image (Grayscale)
        heatmap_img = Image.fromarray(heatmap_uint8, mode='L')

        # Encode as Base64 PNG
        buffered = io.BytesIO()
        heatmap_img.save(buffered, format="PNG")
        heatmap_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return StandardResponse(success=True, endpoint="predict", data={'prediction': predictions, 'gradcam_heatmap_b64': heatmap_b64})

    except Exception as e:
        generic_logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")



@app.get('/health', summary="Check Server Status", response_model=StandardResponse, dependencies=[Depends(validate_api_key)])
async def health():
    """
    Checks API status.
    """
    with model_manager.lock:
        is_ready = model_manager.model is not None

    status = "online" if is_ready else "no_model"
    return StandardResponse(success=True, endpoint="health", data={'API status': status})


# if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run(app, host="0.0.0.0", port=8000)