"""Router for model-related endpoints."""

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from rose_core.config.settings import settings
from rose_server.fs import check_file_path
from rose_server.models.store import (
    create as create_language_model,
    delete as delete_language_model,
    get as get_language_model,
    list_all,
)
from rose_server.schemas.models import CreateModelRequest

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)


@router.get("/models")
async def openai_api_models() -> JSONResponse:
    """OpenAI API-compatible endpoint that lists available models.

    Returns:
        JSON response in OpenAI format with available models
    """
    models = await list_all()
    model_data = []

    # Add language models from database
    for model in models:
        model_data.append(
            {
                "id": model.id,
                "object": "model",
                "created": model.created_at,
                "owned_by": model.owned_by,
                "permission": json.loads(model.permissions) if model.permissions else [],
                "root": model.root or model.id,
                "parent": model.parent,
            }
        )
    openai_response = {"object": "list", "data": model_data}
    return JSONResponse(content=openai_response)


@router.get("/models/{model_id}")
async def get_model_details(model_id: str) -> JSONResponse:
    """Get details about a specific model."""
    model = await get_language_model(model_id)
    if model:
        return JSONResponse(
            content={
                "id": model.id,
                "object": "model",
                "created": model.created_at,
                "owned_by": model.owned_by,
                "permission": json.loads(model.permissions) if model.permissions else [],
                "root": model.root or model.id,
                "parent": model.parent,
                # Extra field for internal use
                "model_name": model.model_name,
                "lora_target_modules": model.get_lora_modules(),
            }
        )
    raise HTTPException(
        status_code=404,
        detail={
            "error": {
                "message": f"The model '{model_id}' does not exist",
                "type": "invalid_request_error",
                "param": None,
                "code": "model_not_found",
            }
        },
    )


@router.post("/models/{model_name}/download")
async def download_model(model_name: str) -> JSONResponse:
    """Pre-download a model to avoid blocking during inference."""
    model = await get_language_model(model_name)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    try:
        return JSONResponse(
            content={
                "status": "downloading",
                "message": f"Model '{model_name}' download started in background",
                "model": model_name,
            }
        )
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download model: {str(e)}")


@router.delete("/models/{model}")
async def delete_model(model: str) -> JSONResponse:
    """Delete a fine-tuned model."""
    model_obj = await get_language_model(model)
    if not model_obj:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"The model '{model}' does not exist",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )

    if not model_obj.is_fine_tuned:
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "message": f"Cannot delete base model '{model}'. Only fine-tuned models can be deleted.",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_deletable",
                }
            },
        )

    # Delete from database
    await delete_language_model(model)

    # Delete model files if they exist
    if model_obj.path:
        model_path = Path(settings.data_dir) / model_obj.path
        if await check_file_path(model_path):
            await asyncio.to_thread(shutil.rmtree, str(model_path))
            logger.info(f"Deleted model files at: {model_path}")

    logger.info(f"Successfully deleted fine-tuned model: {model}")
    return JSONResponse(content={"id": model, "object": "model", "deleted": True})


@router.post("/models", status_code=201)
async def create_model(request: CreateModelRequest) -> Dict[str, Any]:
    """Create a new model configuration."""
    # Check if model already exists
    existing = await get_language_model(request.id)
    if existing:
        raise HTTPException(
            status_code=409,
            detail={
                "error": {
                    "message": f"Model '{request.id}' already exists",
                    "type": "invalid_request_error",
                    "param": "id",
                    "code": "model_exists",
                }
            },
        )

    # Create the model
    model = await create_language_model(
        id=request.id,
        model_name=request.model_name,
        name=request.name,
        temperature=request.temperature,
        top_p=request.top_p,
        memory_gb=request.memory_gb,
        timeout=request.timeout,
        lora_modules=request.lora_target_modules,
        owned_by=request.owned_by,
    )

    logger.info(f"Created model: {model.id} ({model.model_name})")

    # Return the created model in OpenAI format
    return {
        "id": model.id,
        "object": "model",
        "created": model.created_at,
        "owned_by": model.owned_by,
        "permission": json.loads(model.permissions) if model.permissions else [],
        "root": model.root or model.id,
        "parent": model.parent,
    }
