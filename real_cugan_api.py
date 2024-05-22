from fastapi import APIRouter, HTTPException,Request
from pydantic import BaseModel
from real_cugan.cugan import load_super_resolution_model, perform_super_resolution
from loguru import logger
import os
from config import UPLOAD_DIR,PROCESSED_DIR

router = APIRouter()

class SuperResolutionRequest(BaseModel):
    input_path: str
    scale: int
    denoise_level: int
    half: bool
    device: str
    tile: int
    cache_mode: int
    alpha: float

@router.post("/super-resolution")
def super_resolution(request_data: SuperResolutionRequest, request: Request):
    logger.info(f"Request: {request_data}")

    # 加载模型
    model = load_super_resolution_model(
        scale=request_data.scale,
        denoise_level=request_data.denoise_level,
        half=request_data.half,
        device=request_data.device
    )

    try:
        output_file_name = perform_super_resolution(
            model,
            request_data.input_path,
            PROCESSED_DIR,
            tile=request_data.tile,
            cache_mode=request_data.cache_mode,
            alpha=request_data.alpha
        )


        # 创建处理后的图片的URL
        output_url = f"{request.url.scheme}://{request.url.netloc}/processed/{output_file_name[0]}"

        # 销毁模型
        del model

        return {
            "code": 20000,
            "message": "Image(s) processed successfully.",
            "data": {"output_url": output_url}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 启动 FastAPI 服务时不需要做任何模型加载
