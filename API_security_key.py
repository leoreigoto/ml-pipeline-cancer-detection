"""
This module provides security features for a FastAPI application, specifically for API key validation.
The environment variables are loaded from a `.env` file, ensuring that sensitive information like API keys are not hard-coded into the
application.

Functions:
- validate_api_key: Validates the provided API key against the expected key.
"""

from dotenv import load_dotenv
from fastapi.security import APIKeyHeader
from fastapi import HTTPException, Depends
import os
from pathlib import Path
import secrets

api_key_header = APIKeyHeader(name="UFF-API-KEY", auto_error=False)

# enable the env load when using outside of docker
#BASE_DIR = Path(__file__).resolve().parent
#key_path = BASE_DIR / 'keys' / 'keys_app.env'
#load_dotenv(key_path)

EXPECTED_API_KEY = os.getenv("API_KEY")

async def validate_api_key(api_key: str = Depends(api_key_header)):
    if EXPECTED_API_KEY is None:
        # Fail safe: If server key is missing, deny all access
        raise HTTPException(status_code=500, detail="Server security configuration error")

    if api_key and secrets.compare_digest(api_key, EXPECTED_API_KEY):
        return True
    else:
        raise HTTPException(status_code=403, detail="Invalid API Key")
