"""
Configuration Loader Module for Machine Learning Applications.

This module contains functions for loading and validating configuration settings 
from JSON files for a machine learning application, specifically 
focusing on training phase. It ensures that all necessary configuration 
parameters are present, either by reading them from a file or by setting them to 
default values if they are missing or in case of errors during file loading.

Function:
- load_train_config: Loads and validates the training configuration.
  Reads a configuration file, checks for required fields, and fills in any missing 
  fields with default values. It handles file not found, invalid JSON format, and other unexpected 
  errors by returning default configurations.
  Default configuration doesn't apply for SQL fields: url and query.

Exception handling and logging are integral parts of the module, ensuring that any issues with 
configuration files are clearly reported and gracefully handled.

SQL query and URL dont have a default value, in case they are missing
the code will get an error.
When database is ready, consider adding default values.
"""

from dataclasses import dataclass
from pydantic import BaseModel
from typing import Optional


@dataclass(frozen=True)
class APIConfig:
    #TO_DO enable_pred_data_log = True
    model_update_timer: int = 60*60   # in seconds
    health_check_timer: int = 5*60     # in seconds
    # Overwritten with MLFLOW_TRACKING_URI (configured in the docker container)
    tracking_uri: str ='http://127.0.0.1:5000'


class StandardResponse(BaseModel):
    """
    Defines the structure of a standard response for API endpoints.

    This Pydantic model is used to standardize responses from various API endpoints, ensuring a consistent
    structure across the API. It includes a success status, endpoint details, and an optional data field 
    for additional information or results.

    Attributes:
        success (bool): Indicates if the request was successful.
        endpoint (str): The name or path of the endpoint.
        data (dict, optional): Additional data or results from the endpoint. Default is None.
    """
    success: bool
    endpoint: str
    data: Optional[dict] = None
    


# TO_DO adapt and implement
#def TO_DO_get_predict_log(input_data: List[InputData], output_data: dict,model_name,
#                    model_version,model_alias=None):
#    """
#    Create and return a JSON-formatted log entry for prediction requests and outputs.
#
#    This function pairs each input data item with its corresponding output prediction and includes
#    details about the model used for the prediction. The result is formatted as a JSON string.
#
#    Args:
#        input_data (List[InputData]): A list of input data instances.
#        output_data (List[OutputData]): A list of output data instances corresponding to the input data.
#        model_name (str): The name of the prediction model.
#        model_version (str): The version of the prediction model.
#
#    Returns:
#        str: A JSON-formatted string summarizing the prediction data and results.
#    """
#    summarized_data = []
#    for inp, out in zip(input_data, output_data):
#        summarized_item = {
#            # Convert input data to a dictionary
#            #"input": inp.dict(),
#            "input": inp.model_dump(),
#            # Extract price from OutputData
#            "predicted_price": out,
#            "model_name": model_name,
#            "model_version": model_version,
#        }
#        summarized_data.append(summarized_item)
#
#    return json.dumps(summarized_data, indent=2)  # Convert to a JSON string for logging