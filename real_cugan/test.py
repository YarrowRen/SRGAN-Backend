from real_cugan.cugan import load_super_resolution_model,perform_super_resolution
import os

# 获取文件 B 所在的目录
dir_path = os.path.dirname(os.path.realpath(__file__))

# 基于文件 B 的位置构建输入和输出路径
input_path = os.path.join(dir_path, 'input_dir')
output_path = os.path.join(dir_path, 'output_dir')


# 使用示例
model = load_super_resolution_model(scale=2, denoise_level=3, half=True, device="cuda:0")
perform_super_resolution(model, input_path, output_path, tile=5, cache_mode=1, alpha=1.0)