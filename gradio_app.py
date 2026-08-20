
import base64
import cv2
from dotenv import load_dotenv
import gradio as gr
import io
import numpy as np
import os
from pathlib import Path
from PIL import Image
import requests


# URL to predict endpoint
# This prioritizes the Docker internal name (api), but defaults to localhost if running outside Docker
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict_with_gradcam")

# Load the UFF logo
uff_logo_path = "file/uff_logo.png"

# enable the env load when using outside of docker
#key_path =  Path('client_key') /'client_keys.env'
#load_dotenv(key_path)
#api_key = os.getenv('APP_Key_5839123')
api_key= os.getenv('API_KEY')
header_UFF_API = {'UFF-API-KEY': api_key}


def image_to_base64_uri_png_only(path_str):
    """
    Encodes an image file to a Base64 data URI,
    assuming the file is always a PNG.
    """
    path = Path(path_str)
    if not path.is_file():
        print(f"Warning: Image file not found at {path.resolve()}")
        return None
    try:
        # Hardcode the MIME type for PNG
        mime_type = "image/png"

        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        print(f"Successfully encoded {path} as PNG Base64 URI.")
        # Construct the data URI with the hardcoded MIME type
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        print(f"Error encoding image {path} to Base64: {e}")
        return None


def numpy_overlay_heatmap(original_pil_image, heatmap_b64, alpha=0.5):
    """
    Merges the original breast image with the heatmap received from the API.
    """
    original_np = np.array(original_pil_image)
    heatmap_bytes = base64.b64decode(heatmap_b64)
    heatmap_pil = Image.open(io.BytesIO(heatmap_bytes))
    heatmap_np = np.array(heatmap_pil)

    # OpenCV resize expects (Width, Height), shape is (Height, Width)
    dsize = (original_np.shape[1], original_np.shape[0])
    heatmap_resized = cv2.resize(heatmap_np, dsize, interpolation=cv2.INTER_LINEAR)

    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    # Convert Heatmap BGR (OpenCV standard) to RGB (PIL/Gradio standard)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_np, 1 - alpha, heatmap_colored, alpha, 0)
    return Image.fromarray(overlay)

def predict_cancer(image, request: gr.Request):
    """
    Accepts an uploaded image, sends it to the prediction API, and returns
    the prediction class, confidence, and a Grad-CAM image.
    """
    if image is None:
        return None, "No Image", "N/A"
    username = request.username if request.username else "Unknown"
    print(f"Request made by user: {username}")
    
    try:
        # Convert the image to a base64-encoded string for json compatibility
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

        payload = {"inputs": encoded_image}
        print('sending request')
        response = requests.post(API_URL, json=payload, headers=header_UFF_API)
        response.raise_for_status()  # Raise an error for HTTP status codes 4xx/5xx

        api_response = response.json()

        if not api_response.get("success", False):
            raise ValueError(f"API returned failure: {api_response}")

        data = api_response.get("data", {})
        raw_prediction = data.get("prediction")
        # The API returns a list because it supports a batch of images.
        if isinstance(raw_prediction, list):
            raw_prediction = raw_prediction[0]
        else:
            raw_prediction = raw_prediction
        gradcam_b64 = data.get("gradcam_heatmap_b64")

        if raw_prediction is None or gradcam_b64 is None:
            raise ValueError("API response missing 'prediction' or 'gradcam_heatmap_b64'")
        if raw_prediction >= 0.5:
            prediction_class = "Positive (Possible Abnormality)" # Stands for sick class
            confidence = raw_prediction
        else:
            prediction_class = "Negative (Healthy)"
            confidence = 1 - raw_prediction
        
        # Format confidence as a percentage string
        confidence_str = f"{confidence:.2%}"

        final_overlay_image = numpy_overlay_heatmap(image, gradcam_b64, alpha=0.4)
        return final_overlay_image, prediction_class, confidence_str

    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {e}")
        return None, "API Request Error", "Check Server Status"
    except Exception as e:
        print(f"Processing Error: {e}")
        return None, f"Error: {str(e)}", "N/A"


uff_logo_data_uri = image_to_base64_uri_png_only(uff_logo_path)
logo_html = f'<div style="text-align: center;"><img src="{uff_logo_data_uri}" alt="UFF Logo" style="width: 150px; margin-bottom: 20px;"></div>' if uff_logo_data_uri else ""

description = f"""
{logo_html}
<h3>Breast Cancer Thermography Prediction</h3>
<p>This tool is designed for medical professionals to assist in the assessment of breast thermography images.</p>
<p><strong>Usage:</strong> Upload a high-quality thermography image of the breast, and the system will provide a preliminary prediction to help guide further diagnostics.</p>
"""

gr_interface = gr.Interface(
    fn=predict_cancer,
    inputs=gr.Image(type="pil", label="Upload Breast Thermography Image"),
    outputs=[
        gr.Image(type="pil", label="Grad-CAM Analysis (Overlay"),
        gr.Textbox(label="Prediction Class"),
        gr.Textbox(label="Confidence"),
    ],
    title="Breast Cancer Prediction - UFF",
    description=description,
    theme="default",
    css="""
        body { background-color: #111111 !important; }
        .gradio-container {
            font-family: 'Arial', sans-serif;
            background-color: #000000 !important;
            color: #f0f0f0 !important;
            border: 1px solid #444;
            padding: 20px;
            max-width: 800px;
            margin: auto;
            border-radius: 10px;
        }
        .gradio-container h3 { color: #ffffff !important; }
          .gradio-container p { color: #f0f0f0 !important; 
              line-height: 1.5;
        }
        .gradio-container strong {
              color: #ffffff !important;
        }
          /* Style input/output labels */
        label span {
            color: #cccccc !important;
        }
        .gr-button-primary, button.primary {
            background-color: #0056b3 !important;
            color: white !important;
            border: none !important;
        }
        .gr-button-primary:hover, button.primary:hover {
            background-color: #003d82 !important;
        }
        .gradio-container p[style*="font-size: 12px"] {
            color: #cccccc !important;
        }
    """,
    flagging_mode="manual",
    flagging_options=["incorrect","needs_review"],
)

# mock a db
users_db = {
    "doctor1": "password1",
    "doctor2": "password2",
    "doctor3": "password3",
}

# Custom authentication logic
def authenticate_user(username, password):
    if users_db.get(username) == password:
        return True
    else:
        return False

if __name__ == "__main__":
    gr_interface.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True, # Set to True for public link
    inbrowser=False,
    height=800, # 600 / 800
    auth=authenticate_user
    )
