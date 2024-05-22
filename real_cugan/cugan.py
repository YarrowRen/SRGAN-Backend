import cv2, traceback
import os

import numpy as np

from .upcunet_v3 import RealWaifuUpScaler
from loguru import logger
import time
from torch.utils.tensorboard import SummaryWriter

# 获取当前文件（即__init__.py）的绝对路径
package_dir = os.path.dirname(os.path.abspath(__file__))

# 设置权重目录为当前文件所在目录下的 weights_v3 文件夹
WEIGHTS_DIR = os.path.join(package_dir, 'weights_v3')

def get_model_path(scale, denoise_level):
    """
    根据超分倍率和降噪等级获取模型路径。

    :param scale: 超分倍率。
    :param denoise_level: 降噪等级。
    :return: 对应的模型文件路径。
    """
    if denoise_level == 0:
        model_name = f'up{scale}x-latest-no-denoise.pth'
    elif denoise_level == 4:
        model_name = f'up{scale}x-latest-conservative.pth'
    else:
        model_name = f'up{scale}x-latest-denoise{denoise_level}x.pth'
    return os.path.join(WEIGHTS_DIR, model_name)

def load_and_perform_super_resolution(input_path, output_dir, scale, tile, cache_mode, alpha, half, device, denoise_level):
    """
    对指定路径的图像执行超分辨率处理。

    :param input_path: 输入文件或文件夹的路径。
    :param output_dir: 输出文件夹的路径。
    :param scale: 超分倍率。
    :param tile: 平铺模式参数。
    :param cache_mode: 缓存模式参数。
    :param alpha: alpha 参数。
    :param half: 是否使用半精度。
    :param device: 使用的设备（'cpu' 或 'cuda'）。
    :param denoise_level: 降噪等级。
    """
    logger.info("Starting super resolution process...")


    try:
        # 创建输出文件夹
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory '{output_dir}' created.")

        # 获取模型路径
        model_path = get_model_path(scale, denoise_level)
        logger.info(f"Using model at path: {model_path}")

        # 加载模型
        logger.info("Loading upscaling model...")
        upscaler = RealWaifuUpScaler(scale, model_path, half, device)

        # 检查输入路径是文件还是文件夹
        if os.path.isdir(input_path):
            files = [os.path.join(input_path, f) for f in os.listdir(input_path)]
            logger.info(f"Processing all files in directory: {input_path}")
        else:
            files = [input_path]
            logger.info(f"Processing single file: {input_path}")

        # 处理每个文件
        for file in files:
            logger.info(f"Upscaling file: {file}")
            _, ext = os.path.splitext(file)
            frame = cv2.imread(file)[:, :, [2, 1, 0]]
            result = upscaler(frame, tile, cache_mode=cache_mode, alpha=alpha)[:, :, ::-1]

            # 创建输出文件路径
            output_file = os.path.basename(file)
            output_file = os.path.splitext(output_file)[0] + f"_scale{scale}_tile{tile}_cache{cache_mode}_alpha{alpha}_denoise{denoise_level}{ext}"
            output_path = os.path.join(output_dir, output_file)

            # 保存结果
            cv2.imwrite(output_path, result)
            logger.info(f"Saved upscaled image to: {output_path}")

    except Exception as e:
        # 打印异常信息
        logger.error("An error occurred during super resolution.")
        traceback.print_exc()



def load_super_resolution_model(scale, denoise_level, half, device):
    """
    加载超分辨率模型。

    :param scale: 超分倍率。
    :param denoise_level: 降噪等级。
    :param half: 是否使用半精度。
    :param device: 使用的设备（'cpu' 或 'cuda'）。
    :return: 加载的模型。
    """
    logger.info("Loading super resolution model...")
    model_path = get_model_path(scale, denoise_level)
    logger.info(f"Model path: {model_path}")
    model = RealWaifuUpScaler(scale, model_path, half, device)

    return model

def perform_super_resolution(model, input_path, output_dir, tile, cache_mode, alpha):
    """
    使用加载好的模型对指定路径的图像执行超分辨率处理。

    :param model: 加载好的模型。
    :param input_path: 输入文件或文件夹的路径。
    :param output_dir: 输出文件夹的路径。
    :param tile: 平铺模式参数。
    :param cache_mode: 缓存模式参数。
    :param alpha: alpha 参数。
    """
    logger.info("Starting super resolution process...")
    output_file_name = []  # 用于存储输出文件名称

    logger.info(f"Output directory: {output_dir}")  # 添加这行来打印输出目录
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory created at: {output_dir}")

        if os.path.isdir(input_path):
            files = [os.path.join(input_path, f) for f in os.listdir(input_path)]
            logger.info(f"Processing all files in directory: {input_path}")
        else:
            files = [input_path]
            logger.info(f"Processing single file: {input_path}")

        for file in files:
            logger.info(f"Upscaling file: {file}")
            _, ext = os.path.splitext(file)
            frame = cv2.imread(file)[:, :, [2, 1, 0]]
            result = model(frame, tile, cache_mode=cache_mode, alpha=alpha)[:, :, ::-1]
            timestamp = time.strftime("%Y%m%d%H%M%S")

            output_file = os.path.basename(file)
            output_file = os.path.splitext(output_file)[0] + f"_{timestamp}{ext}"
            output_path = os.path.join(output_dir, output_file)

            cv2.imwrite(output_path, result)
            logger.info(f"Upscaled image saved to: {output_path}")
            # 将完整的输出路径添加到列表中
            output_file_name.append(output_file)

    except Exception as e:
        logger.error("An error occurred during super resolution.")
        traceback.print_exc()
    return output_file_name

def perform_super_resolution_get_result(model, input_path, output_dir, tile, cache_mode, alpha):
    """
    使用加载好的模型对指定路径的图像执行超分辨率处理。

    :param model: 加载好的模型。
    :param input_path: 输入文件或文件夹的路径。
    :param output_dir: 输出文件夹的路径。
    :param tile: 平铺模式参数。
    :param cache_mode: 缓存模式参数。
    :param alpha: alpha 参数。
    """
    logger.info("Starting super resolution process...")
    output_file_name = []  # 用于存储输出文件名称

    logger.info(f"Output directory: {output_dir}")  # 添加这行来打印输出目录
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory created at: {output_dir}")

        if os.path.isdir(input_path):
            files = [os.path.join(input_path, f) for f in os.listdir(input_path)]
            logger.info(f"Processing all files in directory: {input_path}")
        else:
            files = [input_path]
            logger.info(f"Processing single file: {input_path}")

        for file in files:
            logger.info(f"Upscaling file: {file}")
            _, ext = os.path.splitext(file)
            frame = cv2.imread(file)[:, :, [2, 1, 0]]
            result = model(frame, tile, cache_mode=cache_mode, alpha=alpha)[:, :, ::-1]
            timestamp = time.strftime("%Y%m%d%H%M%S")

            output_file = os.path.basename(file)
            output_file = os.path.splitext(output_file)[0] + f"_{timestamp}{ext}"
            output_path = os.path.join(output_dir, output_file)

            cv2.imwrite(output_path, result)
            logger.info(f"Upscaled image saved to: {output_path}")
            # 将完整的输出路径添加到列表中
            output_file_name.append(output_file)

    except Exception as e:
        logger.error("An error occurred during super resolution.")
        traceback.print_exc()
    return result


# 使用示例
# model = load_super_resolution_model(scale=2, denoise_level=2, half=False, device="cpu")
# perform_super_resolution(model=model, input_path="path/to/input", output_dir="path/to/output", tile=2, cache_mode=1, alpha=1.0)