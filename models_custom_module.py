import cv2
import lightning.pytorch as pl
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pickle
import torch
from torchmetrics import MeanMetric, ConfusionMatrix
from torchmetrics.classification import (
    BinaryPrecision,
    BinaryRecall,
    BinaryAccuracy,
    BinaryF1Score,
    BinarySpecificity
)
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Any, Optional, Dict, Type


def load_model_and_pipeline(model_uri: str, pipeline_uri: str) -> Tuple[torch.nn.Module, Any]:
    """
    Downloads artifacts and loads the model and pipeline objects.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = mlflow.pytorch.load_model(
        model_uri,
        map_location=torch.device(device)
    )

    pipeline_path = mlflow.artifacts.download_artifacts(pipeline_uri)
    with open(pipeline_path, "rb") as file:
        pipeline = pickle.load(file)

    return model, pipeline

def log_confusion_matrix(current_epoch, conf_matrix):
    fig = Figure(figsize=(10, 10))
    ax = fig.subplots()
    cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)

    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")

    fig.colorbar(cax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f'Confusion Matrix Epoch {current_epoch}')
    ax.set_xticklabels([''] + ['Negative', 'Positive'])
    ax.set_yticklabels([''] + ['Negative', 'Positive'])

    mlflow.log_figure(fig, f"confusion_matrix/epoch_{current_epoch}.png")
    plt.close(fig)

def overlay_heatmap(image: torch.Tensor, heatmap: torch.Tensor):
    """
    Overlays the heatmap on the original image.

    Args:
        image: Tensor [C, H, W] (normalized)

        heatmap: Tensor [1, H, W] or [H, W]
    """
    # Unnormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_unnorm = image.clone().cpu() * std + mean
    img_np = img_unnorm.permute(1, 2, 0).detach().numpy()
    img_np = np.clip(img_np, 0, 1)
    img_uint8 = (img_np * 255).astype(np.uint8)

    heatmap_np = heatmap.detach().cpu().numpy()
    heatmap_resized = cv2.resize(heatmap_np, (img_uint8.shape[1], img_uint8.shape[0]))

    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Overlay heatmap
    superimposed_img = cv2.addWeighted(img_uint8, 0.6, heatmap_color, 0.4, 0)

    return superimposed_img

# This allows to dynamically add new models without changing implementation code
# Just add the decorator to new model classes
MODEL_REGISTRY: Dict[str, Type[pl.LightningModule]] = {}
def register_model(name: str):
    """Decorator to register models automatically."""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

@register_model("EffnetB0")
class EffnetB0(pl.LightningModule):
    def __init__(
            self,
            model_name="EfficientnetB0",
            learning_rate=0.01
    ):
        super().__init__()

        self.save_hyperparameters()

        effnet = models.efficientnet_b0(weights="IMAGENET1K_V1")

        for param in effnet.parameters():
            param.requires_grad = False

        last_layer_in = effnet.classifier[1].in_features
        effnet.classifier[1] = nn.Linear(last_layer_in, 1, bias=True)

        self.model = effnet

        # Target layer for GradCAM (Last convolutional block)
        self.gradcam_layer = self.model.features[8]

        self.mean_train_loss = MeanMetric()
        self.mean_valid_loss = MeanMetric()
        self.mean_train_acc = BinaryAccuracy()
        self.mean_valid_acc = BinaryAccuracy()

        self.train_precision = BinaryPrecision()
        self.train_recall = BinaryRecall()
        self.validation_precision = BinaryPrecision()
        self.validation_recall = BinaryRecall()

        self.train_f1 = BinaryF1Score()
        self.train_specificity = BinarySpecificity()
        self.validation_f1 = BinaryF1Score()
        self.val_specificity = BinarySpecificity()

        self.valid_confusion_matrix = ConfusionMatrix(num_classes=2, task='binary')

    def forward(self, x):
        return self.model(x)

    def forward_with_gradcam_heatmap(self,x):
        """Generates the Grad-CAM heatmap and the model's prediction."""
        if x.size(0) != 1:
            raise ValueError("Grad-CAM method currently supports only batch_size=1")

        # Local storage for hooks
        activations = []
        gradients = []

        # Register hooks
        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        forward_handle = self.gradcam_layer.register_forward_hook(forward_hook)
        backward_handle = self.gradcam_layer.register_full_backward_hook(backward_hook)

        # Forward pass
        # x needs to have gradient enabled, otherwise torch.set_grad_enabled bellow wont work.
        if not x.requires_grad:
            x.requires_grad_(True)
        try:
            # Enable grad temporarily even if model is in eval mode
            with torch.set_grad_enabled(True):
                logits = self.model(x)
                #Backward pass
                self.model.zero_grad()
                # Assuming batch_size=1.
                logits.backward(retain_graph=False)

            # Get activations and gradients
            activation = activations[0].detach()
            gradient = gradients[0].detach()

            # Standard Grad-CAM calculation
            weights = torch.mean(gradient, dim=[2, 3], keepdim=True)
            heatmap = torch.sum(weights * activation, dim=1, keepdim=True)
            heatmap = F.relu(heatmap)
            # Normalize safely to avoid div by 0
            heatmap /= torch.max(heatmap + 1e-8)
            return logits, heatmap

        # This ensures hooks are always removed, even if the code above fails
        finally:
            forward_handle.remove()
            backward_handle.remove()

    def training_step(self, batch, batch_idx):
        data, target = batch

        output = self(data).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(output, target.float())

        self.mean_train_loss(loss)

        # Calculate batch predictions
        pred_batch = torch.sigmoid(output) > 0.5  # Convert to binary predictions

        # Update and log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.mean_train_acc(pred_batch, target)
        self.log('train_acc', self.mean_train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.train_precision(pred_batch, target)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.train_recall(pred_batch, target)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.train_f1(pred_batch, target)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.train_specificity(pred_batch, target)
        self.log('train_specificity', self.train_specificity, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, *args, **kwargs):
        data, target = batch

        output = self(data).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(output, target.float())

        pred_batch = torch.sigmoid(output) > 0.5
        self.valid_confusion_matrix.update(pred_batch.int(), target.int())

        # Update and log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.mean_valid_acc(pred_batch, target)
        self.log('val_acc', self.mean_valid_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.validation_precision(pred_batch, target)
        self.log('val_precision', self.validation_precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.validation_recall(pred_batch, target)
        self.log('val_recall', self.validation_recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.validation_f1(pred_batch, target)
        self.log('val_f1', self.validation_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.val_specificity(pred_batch, target)
        self.log('val_specificity', self.val_specificity, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        conf_matrix = self.valid_confusion_matrix.compute().cpu().numpy()
        tn, fp, fn, tp = conf_matrix.ravel()
        print(f"True Positives: {tp}, False Negatives: {fn}, True Negatives: {tn}, False Positives: {fp} ")
        log_confusion_matrix(self.current_epoch, conf_matrix)
        self.valid_confusion_matrix.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


class ModelWithPipeline(nn.Module):
    """
    Wrapper to bundle the model with its preprocessing pipeline for easier inference.

    This class serves as a container that can optionally handle raw inputs (like PIL images)
    by automatically applying the associated transformation pipeline before passing the
    data to the model.

    Args:
        model (nn.Module): The underlying PyTorch model.
        pipeline (Any): The preprocessing pipeline.
        smart_apply_pipeline (bool, optional): Controls the inference behavior.
            - If True: The forward method checks if the input is NOT a Tensor (e.g., a PIL Image).
                If it detects a raw input, it applies the pipeline and adds a batch dimension.
            - If False (default): The pipeline is always included in the forward pass.
                The model assumes the input `x` is raw (e.g., a PIL Image) and NOT yet pre-processed.
        """

    def __init__(self, model: nn.Module, pipeline: Any, smart_apply_pipeline:bool = False):
        super().__init__()
        self.model = model
        self.pipeline = pipeline
        self.smart_apply_pipeline = smart_apply_pipeline
        self.eval()

    def _preprocess(self, x):

        device = next(self.model.parameters()).device

        should_apply_pipeline = True

        if self.smart_apply_pipeline and isinstance(x, torch.Tensor):
            should_apply_pipeline = False

        if should_apply_pipeline:
            x = self.pipeline(x)
            # Add batch dimension [1, C, H, W]
            x = x.unsqueeze(0)

        return x.to(device)

    def forward(self, x):
        """Pre-process input and call the forward method."""
        x = self._preprocess(x)
        return self.model(x)

    def forward_with_gradcam_heatmap(self,x):
        """Pre-processes input and calls the model's Grad-CAM method."""
        # Check if the underlying model supports Grad-CAM
        if hasattr(self.model, "forward_with_gradcam_heatmap"):
            x = self._preprocess(x)
            return self.model.forward_with_gradcam_heatmap(x)
        else:
            raise NotImplementedError("The underlying model does not support forward_with_gradcam_heatmap")


def get_train_model(
        model_name: str,
        logger,
        model_params: Optional[Dict[str, Any]] = None,
        checkpoint: str = None
) -> pl.LightningModule:
    """
    Factory function to instantiate models dynamically.

    Args:
        model_name: Name of the model registered in MODEL_REGISTRY.
        logger: Logger for error handling.
        model_params: Dictionary of parameters to pass to the model __init__ (e.g. {'learning_rate': 0.001}).
        checkpoint: Path to a .ckpt file to load weights from.
    """
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        logger.error(f"Model '{model_name}' not found. Available models: {available}")
        raise ValueError(f"Model '{model_name}' not registered.")

    model_cls = MODEL_REGISTRY[model_name]
    params = model_params or {}

    # Logic: Load from checkpoint OR Initialize fresh
    if not checkpoint:
        logger.info(f"Initializing fresh {model_name} with params: {params}")
        return model_cls(**params)
    if checkpoint.endswith(".ckpt"):
        logger.info(f"Loading {model_name} from checkpoint: {checkpoint}")
        # strict=False is useful if new layers are added (like the GradCAM hooks/state)
        # that weren't in the saved checkpoint
        return model_cls.load_from_checkpoint(checkpoint, strict=False, **params)
    elif checkpoint.endswith(".pth") or checkpoint.endswith(".pt"):
        logger.info(f"Loading raw PyTorch weights: {checkpoint}")
        model = model_cls(**params)
        state_dict = torch.load(checkpoint, map_location=model.device)
        # We use strict=False to be safe against missing keys (like if new metrics were added)
        keys = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded weights with keys: {keys}")
        logger.info(f"Missing keys: {keys.missing_keys}")
        logger.info(f"Unexpected keys: {keys.unexpected_keys}")
        return model
    else:
        raise ValueError(f"Unsupported checkpoint extension: {checkpoint}")