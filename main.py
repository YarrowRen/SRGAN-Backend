import shutil

from fastapi import FastAPI, Form, HTTPException,UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from manga_ocr import MangaOcr
from fastapi.middleware.cors import CORSMiddleware
from real_cugan_api import router as cugan_router
from ocr_api import router as ocr_router
from loguru import logger
from uuid import uuid4
import os
from config import UPLOAD_DIR,PROCESSED_DIR

app = FastAPI()
# 假设处理后的文件存储在 'processed_files' 目录
app.mount("/processed", StaticFiles(directory=PROCESSED_DIR), name="processed")
app.mount("/upload", StaticFiles(directory=UPLOAD_DIR), name="upload")
# 将 OCR 路由器挂载到主应用
app.include_router(ocr_router, prefix="/ocr")
app.include_router(cugan_router, prefix="/cugan")

@app.on_event("startup")
async def startup_event():
    logger.info("Main application startup.")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有域，为了安全，您应该在生产中限制特定域
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)


@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    file_ext = os.path.splitext(file.filename)[1]
    file_name = str(uuid4()) + file_ext
    file_path = os.path.join(UPLOAD_DIR, file_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    url = f"{request.url.scheme}://{request.url.netloc}/upload/{file_name}"

    return {"code": 20000, "data": {"path": file_path, "url": url}}



@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
