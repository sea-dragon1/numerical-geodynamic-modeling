# -*- coding: utf-8 -*-

# 绘制一条线上的strain-rate随时间变化

"""
Spacetime plot of stress along a horizontal line (y=const, z=0)
x: horizontal distance (km)
t: time (Ma, upward)
color: stress magnitude (Pa)
"""

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path
import os
import xml.etree.ElementTree as ET
import pandas as pd
from fields import ASPECT_FIELDS, get_field_info

# ================== 2. PVD 加载器 ==================
def load_pvd(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    collection = root.find('Collection')
    files = []
    for dataset in collection:
        t = float(dataset.attrib['timestep'])
        fname = dataset.attrib['file']
        files.append((t, fname))
    return sorted(files, key=lambda x: x[0])


def plot_spacetime_line(pvd_file, field_name, x_min, x_max, y_line, z_line, n_x, save_dir):
    
    # ================== 3. 提取时空数据 ==================
    print("Loading PVD files...")
    time_files = load_pvd(pvd_file)
    n_times = len(time_files)
    print(f"Found {n_times} time steps.")

    # 预分配数组
    x_coords = np.linspace(x_min, x_max, n_x)
    stress_2d = np.full((n_times, n_x), np.nan)   # [time, x]
    time_ma_list = []

    print(f"Extracting {field_name} along line...")
    for t_idx, (t_yr, rel_path) in enumerate(time_files):
        pvtu = pvd_file.parent / rel_path
        reader = pv.get_reader(pvtu)
        mesh = reader.read()

        time_ma = t_yr / 1e6
        time_ma_list.append(time_ma)

        # 构建采样线
        line = pv.Line(pointa=(x_min, y_line, z_line), pointb=(x_max, y_line, z_line), resolution=n_x-1)
        sampled = mesh.sample_over_line(
            pointa=(x_min, y_line, z_line),
            pointb=(x_max, y_line, z_line),
            resolution=n_x - 1
        )

        # 提取应力（取第二不变量）
        stress_vec = sampled.point_data[field_name]
        stress_scalar = stress_vec[:, 0] if stress_vec.ndim > 1 else stress_vec

        # 填充（补齐长度）
        stress_2d[t_idx, :len(stress_scalar)] = stress_scalar

        if (t_idx + 1) % max(1, n_times//10) == 0:
            print(f"  Processed {t_idx+1}/{n_times} steps...")

    time_ma_array = np.array(time_ma_list)

    # 保存为 CSV（可选）
    df = pd.DataFrame(stress_2d, index=time_ma_array, columns=x_coords/1e3)
    df.to_csv(save_dir / f'{field_name}_spacetime_line.csv')
    print(f"Data saved to CSV: {save_dir / f'{field_name}_spacetime_line.csv'}")

    # ================== 4. 绘图（论文级） ==================
    plt.figure(figsize=(10, 6), dpi=150)

    # 网格：X (km), T (Ma)
    X_grid, T_grid = np.meshgrid(x_coords/1e3, time_ma_array)

    # 绘图：pcolormesh
    vmin, vmax = np.nanpercentile(stress_2d, [5, 99])  # 自动裁剪极端值
    im = plt.pcolormesh(X_grid, T_grid, stress_2d,
                        cmap='Reds', shading='auto',
                        vmin=vmin, vmax=vmax)

    # --- 在 plt.pcolormesh() 之后添加 ---
    ax = plt.gca()

    # 关键时间点列表（可自定义）
    key_times = [
        {"time": 10.0, "label": "Extension onset"},
        {"time": 13.0, "label": "Extension peak"},
        {"time": 15.0, "label": "Subduction initiation"}
    ]
    # 绘制水平线 + 文字
    for item in key_times:
        t_ma = item["time"]
        label = item["label"]
        
        # 水平线
        ax.axhline(y=t_ma, color='black', linestyle='-', linewidth=1.2, alpha=0.8)
        
        # 文字：放在线的左上方（x 靠左，y 略高）
        ax.text(
            x=ax.get_xlim()[0],           # x = 图左边界
            y=t_ma + 0.3,                 # y 向上偏移 0.3 Ma（可调）
            s=label,
            fontsize=10,
            color='black',
            fontweight='bold',
            va='bottom',                  # 文字底部对齐线
            ha='left',                    # 文字左对齐
            transform=ax.transData        # 使用数据坐标
        )
    # 色标
    cbar = plt.colorbar(im, label=f'{field_name} ({info.unit})', extend='both')
    cbar.ax.ticklabel_format(style='sci', scilimits=(0,0))

    # 坐标轴
    plt.xlabel('Distance along line (km)', fontsize=12)
    plt.ylabel('Time (Ma)', fontsize=12)
    plt.title(f'{field_name} evolution along y = {y_line/1e3:.1f} km, z = {z_line/1e3:.0f} km', fontsize=13)

    # 刻度
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.invert_yaxis()  # 时间从下到上递增（0 Ma 在下）

    # 美化
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # 保存
    save_path = save_dir / f'{field_name}_spacetime_plot.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {save_path}")

    plt.show()

# ================== 1. 参数设置 ==================
path = Path(r'E:\backup\DoubleSubduction\model\double_plstc_subeen\gwb_add\ts\thickness_cftold\t70\t80t70\output_t80t70_cftold')
pvd_file = path / 'solution.pvd'
# stress strain_rate
field_name = 'strain_rate'                     # 应力第二不变量
info = get_field_info(field_name)
print(f"Extracting field: {field_name} ({info.plot_quantity})")

y_line = 641643.84375                     # 固定 y
z_line = 0.0                              # 固定 z

# x 范围（建议覆盖模型主要区域）
x_min, x_max = 2000000.0, 3000000.0       # 单位：m，改为你的模型范围
n_x = 200                                 # x 方向采样点数（分辨率）

# 输出设置
save_dir = path / 'post_processing'
save_dir.mkdir(exist_ok=True)

plot_spacetime_line(pvd_file, field_name, x_min, x_max, y_line, z_line, n_x, save_dir)

