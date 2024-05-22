import torch
import time

def benchmark_device(device):
    # 设置矩阵大小，例如：1024x1024
    size = 1024

    # 创建两个随机矩阵
    mat1 = torch.rand(size, size, device=device)
    mat2 = torch.rand(size, size, device=device)

    # 预热，避免启动CUDA等的开销影响时间测量
    for _ in range(10):
        _ = torch.mm(mat1, mat2)

    # 清空CUDA核心和缓存以确保更准确的时间测量
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    start_time = time.time()

    # 执行100次矩阵乘法以获取平均运行时间
    for _ in range(100):
        _ = torch.mm(mat1, mat2)

    # 确保所有CUDA核心完成计算
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    end_time = time.time()

    print(f"Average time on {device}: {(end_time - start_time) / 100:.6f} seconds per multiplication")


# 检测并使用可用的设备：MPS（苹果M系列芯片）或CUDA（NVIDIA GPU），否则使用CPU
device = torch.device( 'cuda' )
benchmark_device(device)