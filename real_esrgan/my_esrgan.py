import torch
from PIL import Image
import os
from RealESRGAN import RealESRGAN

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

# 实例化模型并加载权重
model = RealESRGAN(device, scale=2)
model.load_weights('weights/RealESRGAN_x2.pth', download=True)

# 指定输入和输出文件夹
input_folder = 'inputs/'  # 输入文件夹
output_folder = 'results/'  # 输出文件夹

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有图片文件
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
        path_to_image = os.path.join(input_folder, filename)
        image = Image.open(path_to_image).convert('RGB')

        # 使用模型进行超分辨率处理
        sr_image = model.predict(image)

        # 构建输出文件路径
        output_file_path = os.path.join(output_folder, f'sr_{filename}')

        # 保存超分辨率图像
        sr_image.save(output_file_path)

print("All images have been processed.")
