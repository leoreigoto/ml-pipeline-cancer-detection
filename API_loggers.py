"""
This module provides the functionality to create and configure specialized loggers for different 
aspects of an API module within a ML prediction application.

The primary function `get_api_loggers` sets up two distinct loggers: one for general logging
(generic_logger) and one for logging prediction
histories (pred_logger).

Functions:
    get_api_loggers: Initializes and configures three distinct loggers for various logging needs 
                     within the API module.
"""

import logging
#custom imports
from create_logger import get_logger

    
def get_api_loggers(module_name):
    """
    Create and configure loggers for different aspects of an API module.

    This function initializes  distinct loggers for general information,
    and prediction history, respectively.

    Args:
        module_name (str): The name of the module for which loggers are being created. This name is 
                           used as part of the logger's identifier and name.

    Returns:
        tuple: A tuple containing three loggers: generic_logger and pred_logger.
    """
    try:
        # Generic logger for general information
        generic_logger_save_ID=module_name
        generic_logger_name=module_name
        generic_logger_level=logging.INFO
        generic_logger = get_logger(generic_logger_save_ID,generic_logger_name,generic_logger_level)

        # Logger for prediction history (conditional on enable_pred_data_log)
        pred_logger_save_ID=f"{module_name}_preds_history"
        pred_logger_name=f"{module_name}_preds_history"
        pred_logger_level=logging.INFO
        pred_logger = get_logger(pred_logger_save_ID,pred_logger_name,pred_logger_level)
        
    except Exception as e:
        # Handle any exception that occurs during logger creation
        raise RuntimeError(f"Error creating loggers for module {module_name}: {e}")
    
    return generic_logger,pred_logger
