# Standard library imports
import time
import math

# Third-party imports
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Local imports
from config import UPLOAD_DIR, PROCESSED_DIR
from real_cugan.cugan import load_super_resolution_model, perform_super_resolution
from RealESRGAN import RealESRGAN
from resize_file import resize_and_save_image

# Initialize device
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

# Load and configure CUGAN model
cugan_model = load_super_resolution_model(
    scale=2,
    denoise_level=3,
    half=True,
    device="cuda:0"
)
# print(cugan_model.model)

# Instantiate and load ESRGAN model weights
esrgan_model = RealESRGAN(device, scale=2)
esrgan_model.load_weights('real_esrgan/weights/RealESRGAN_x2.pth', download=True)
print(esrgan_model.model)

# Resize input image
input_path = "E:\\downloads\\gan_test_2.png"
input_half_path = resize_and_save_image(input_path)

# Define transformation: Convert image to tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image data from [0, 255] to [0, 1] and to Tensor
])

# Time tracking for CUGAN processing
start_time = time.time()
cugan_output_file_name = perform_super_resolution(
    cugan_model,
    input_half_path,
    PROCESSED_DIR,
    tile=5,
    cache_mode=1,
    alpha=1
)
cugan_time = time.time() - start_time

print("Average Time for CUGAN:", cugan_time)

# Load high-resolution and super-resolution images
hr_image = Image.open(input_path).convert('RGB')
sr_image = Image.open("processed_files/" + cugan_output_file_name[0]).convert('RGB')

# Convert images to tensors
tensor_hr_image = transform(hr_image)
tensor_sr_image = transform(sr_image)

# Move tensors to the same device
tensor_hr_image = tensor_hr_image.to(device)
tensor_sr_image = tensor_sr_image.to(device)

# Loss calculation
loss_fn = torch.nn.MSELoss(reduction='mean')
loss = loss_fn(tensor_sr_image, tensor_hr_image)
print(f"Loss for CUGAN: {loss.item()}")

# PSNR calculation
mse = F.mse_loss(tensor_sr_image, tensor_hr_image).item()
psnr = 20 * math.log10(1.0 / math.sqrt(mse))
print(f"PSNR for CUGAN: {psnr}")

# Convert images to numpy arrays
hr_array = np.array(hr_image)
sr_array = np.array(sr_image)

ssim_values = [ssim(hr_array[:,:,i], sr_array[:,:,i], data_range=sr_array.max() - sr_array.min()) for i in range(3)]
average_ssim = np.mean(ssim_values)

print(f"Average SSIM for CUGAN: {average_ssim}")



# 测量 ESRGAN 处理时间
start_time = time.time()
half_image = Image.open(input_half_path).convert('RGB')
sr_image = esrgan_model.predict(half_image)
esrgan_time = time.time() - start_time

print("Average Time for ESRGAN:", esrgan_time)

tensor_sr_image_esr=transform(sr_image)
tensor_sr_image_esr=tensor_sr_image_esr.to(device)
# Loss calculation
loss_esr = loss_fn(tensor_sr_image_esr, tensor_hr_image)
print(f"Loss for ESRGAN: {loss_esr.item()}")

# PSNR calculation
mse_esr = F.mse_loss(tensor_sr_image_esr, tensor_hr_image).item()
psnr_esr = 20 * math.log10(1.0 / math.sqrt(mse_esr))
print(f"PSNR for ESRGAN: {psnr_esr}")

# Convert images to numpy arrays
sr_esr_array = np.array(sr_image)

ssim_values_esr = [ssim(hr_array[:,:,i], sr_esr_array[:,:,i], data_range=sr_esr_array.max() - sr_esr_array.min()) for i in range(3)]
average_ssim_esr = np.mean(ssim_values_esr)

print(f"Average SSIM for ESRGAN: {average_ssim_esr}")