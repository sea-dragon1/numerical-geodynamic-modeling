#　读取topography文件，显示topography的图像
# 2025/03/19, lhl

import numpy as np
import matplotlib.pyplot as plt
import glob

# 读取数据
file = r'D:\THU-loong\科研\动力学数值模拟-lmx\实战\Double Subduction\model\double_plstc_subeen\gwb\PA500km\t80\v2t80\topography\\'

# 定义文件名模式
file_pattern = file + 'topography.*'

# 使用 glob 匹配所有文件
files = sorted(glob.glob(file_pattern))

# 创建一个图形窗口
plt.figure(figsize=(10, 6))

# 遍历所有文件并读取数据
for file in files:
    # 读取数据
    data = np.loadtxt(file, comments='#', delimiter=' ', unpack=True)
    x = data[0]
    y = data[1]
    z = data[2]

    # 绘制散点图
    scatter = plt.scatter(x, y, c=z, cmap='viridis', s=10, alpha=0.5)

# 添加颜色条
plt.colorbar(scatter, label='Topography')

# 设置图表标题和坐标轴标签
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Combined Topography Map')
plt.grid(True)

# 显示图表
plt.show()