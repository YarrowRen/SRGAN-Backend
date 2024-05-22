from fastapi import APIRouter, Form, HTTPException
from PIL import Image
import io
import base64
from manga_ocr import MangaOcr
from loguru import logger

router = APIRouter()
ocr_engine = None

@router.on_event("startup")
async def startup_event():
    global ocr_engine
    try:
        ocr_engine = MangaOcr()
        logger.info("OCR engine initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize OCR engine: {e}")
        raise e

@router.post("/img-detect/")
async def img_detect(base64_str: str = Form(...)):
    if ocr_engine is None:
        raise HTTPException(status_code=500, detail="OCR engine not initialized.")

    try:
        # 解码Base64字符串
        image_data = base64.b64decode(base64_str)
        # 将字节数据转换为PIL图像对象
        image = Image.open(io.BytesIO(image_data))

        # 在这里可以处理图像（比如保存）
        # image.save("your_image.png")

        text = ocr_engine(image)
        return {"code": 20000, "message": f"图像处理成功: {text}",
                "data": text}
        # return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/img-path-detect/")
async def img_path_detect(path:str):
    try:
        # 将字节数据转换为PIL图像对象
        img = Image.open(path)

        # 在这里可以处理图像（比如保存）
        # image.save("your_image.png")
        text = ocr_engine(img)
        return {"message": f"图像处理成功: {text}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

