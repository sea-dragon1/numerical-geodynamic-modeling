

# -*- coding: utf-8 -*-
"""
Extract stress (2nd invariant) at a fixed point from ASPECT .pvtu time series
and plot its evolution in **Ma** (million years) – ready for publication.
"""

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path
import os
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from typing import Optional

@dataclass
class FieldInfo:
    name: str                 # ASPECT 输出字段名
    unit: str                 # 单位（LaTeX 可读）
    is_scalar: bool           # 是否标量
    components: Optional[int] = None   # 张量分量数（None 表示标量）
    plot_quantity: str = ""   # 推荐绘图的标量量（如 |τ_xy|、II）

# ================== 统一字段配置表 ==================
ASPECT_FIELDS = {
    "stress": FieldInfo(
        name="stress",
        unit="Pa",
        is_scalar=True,
        plot_quantity="II (2nd invariant)"
    ),
    "shear_stress": FieldInfo(
        name="shear_stress",
        unit="Pa",
        is_scalar=False,
        components=6,
        plot_quantity="|τ_xy|"   # 2D 俯冲推荐
    ),
    "strain_rate": FieldInfo(
        name="strain_rate",
        unit="s^{-1}",
        is_scalar=False,
        components=6,
        plot_quantity="II (2nd invariant)"
    ),
    "velocity": FieldInfo(
        name="velocity",
        unit="m/s",
        is_scalar=False,
        components=3,
        plot_quantity="speed"
    ),
    "temperature": FieldInfo(
        name="temperature",
        unit="K",
        is_scalar=True
    ),
    "viscosity": FieldInfo(
        name="viscosity",
        unit="Pa·s",
        is_scalar=True
    ),
    "density": FieldInfo(
        name="density",
        unit="kg/m³",
        is_scalar=True
    ),
    "principal_stress": FieldInfo(
        name="principal_stress",
        unit="Pa",
        is_scalar=False,
        components=3,  # σ1, σ2, σ3
        plot_quantity="σ1 (max principal)"
    )
}

# ----------------------------------------------------------------------
# 1. PVD loader (unchanged)
# ----------------------------------------------------------------------
def load_pvd(file_path: Path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    collection = root.find('Collection')
    files = []
    for dataset in collection:
        t = float(dataset.attrib['timestep'])
        fname = dataset.attrib['file']
        files.append((t, fname))
    return sorted(files, key=lambda x: x[0])


# ----------------------------------------------------------------------
# 2. 参数设置
# ----------------------------------------------------------------------
path = Path(r'E:\backup\DoubleSubduction\model\double_plstc_subeen\gwb_add\ts\thickness_cftold\r80\l70\output_l70r80')
pvd_file = path / 'solution.pvd'

field_name = 'shear_stress'                     # ASPECT 默认输出的应力第二不变量 stress,strain_rate,shear_stress
info = ASPECT_FIELDS[field_name]
print(f"Extracting field: {field_name} ({info.plot_quantity})")
x0, y0, z0 = 2680000.0, 641643.84375, 0.0   # 监测点坐标（单位与模型一致）

# ----------------------------------------------------------------------
# 3. 逐帧提取数据
# ----------------------------------------------------------------------
time_vals_yr = []   # 原始时间（年）
point_vals   = []   # 应力值

# for t_yr, rel_path in load_pvd(pvd_file):
#     pvtu = pvd_file.parent / rel_path
#     mesh = pv.read(pvtu)

#     idx = mesh.find_closest_point((x0, y0, z0))
#     val = mesh[field_name][idx, 0]         # 标量值（Pa）
#     time_vals_yr.append(t_yr)
#     point_vals.append(val)

# 替换整个 for 循环
time_vals_ma = []
point_vals   = []

for t_yr, rel_path in load_pvd(pvd_file):
    pvtu = pvd_file.parent / rel_path

    # 关键：只读点数据，不构建完整网格！
    reader = pv.get_reader(pvtu)
    # reader.enable_point_data()           # 确保读取点数据
    reader.read()                        # 不构建 mesh
    data = reader.read()                 # 直接得到 UnstructuredGrid

    # 快速找最近点（比 mesh.find_closest_point 快）
    idx = data.find_closest_point((x0, y0, z0))
    if not info.is_scalar:
        val = data.point_data[field_name][idx][0]  # 取标量
    else:
        val = data.point_data[field_name][idx]  # 取标量

    time_vals_ma.append(t_yr / 1e6)
    point_vals.append(val)

# ----------------------------------------------------------------------
# 4. 单位转换：年 → 百万年 (Ma)
# ----------------------------------------------------------------------
# time_vals_ma = np.array(time_vals_yr)  # 1 Ma = 1e6 yr

# ----------------------------------------------------------------------
# 5. 绘图（论文级美化）
# ----------------------------------------------------------------------
plt.figure(figsize=(7, 4.5), dpi=150)


# 主曲线
plt.plot(time_vals_ma, point_vals,
         linewidth=1.5, color='#d62728', alpha=0.9, label=f'{field_name} (2nd invariant)')

# --- 在 plt.plot() 之后、坐标轴设置之前添加背景 ---
ax = plt.gca()

# 获取 y 轴范围
ymin, ymax = ax.get_ylim()

# 背景填充：y > 0 浅红，y < 0 浅绿
ax.axhspan(0, ymax, facecolor='#ffe6e6', alpha=0.6, zorder=0)  # 浅红
ax.axhspan(ymin, 0, facecolor='#e6f9e6', alpha=0.6, zorder=0)  # 浅绿

# 零线（增强对比）
ax.axhline(0, color='black', linewidth=1.0, alpha=0.7, zorder=1)

# 可选：标注俯冲启动时间（自行修改数值）
subduction_start_ma = 15   # 示例：15 Ma
plt.axvline(x=subduction_start_ma, color='k', linestyle='--', linewidth=1.2,
            label=f'Subduction initiation ({subduction_start_ma} Ma)')

# 坐标轴设置
ax = plt.gca()
ax.set_xlabel('Time (Ma)', fontsize=12)
ax.set_ylabel(f'{field_name} ({info.unit})', fontsize=12)

# X 轴刻度：主刻度 5 Ma，次刻度 1 Ma
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))

# Y 轴使用科学计数法（避免 1e7 这种写法）
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax.yaxis.get_offset_text().set_fontsize(10)

# 网格 & 边框
ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.5)
ax.grid(True, which='minor', linestyle=':',  linewidth=0.4, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 标题（包含监测点坐标，单位 km）
plt.title(f'{field_name} evolution at ({x0/1e3:.0f}, {y0/1e3:.0f}, {z0/1e3:.0f}) km',
          fontsize=13, pad=12)

# 图例
if 'subduction_start_ma' in locals():
    plt.legend(frameon=False, fontsize=10)

plt.tight_layout()

# ----------------------------------------------------------------------
# 6. 保存高分辨率图片
# ----------------------------------------------------------------------
save_dir = path / 'post_processing'
save_dir.mkdir(parents=True, exist_ok=True)

# 文件名包含坐标（km）+ 单位 Ma
save_name = f'{field_name}_evolution_{int(x0/1e3)}km_{int(y0/1e3)}km_Ma.png'
save_path = save_dir / save_name

plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {save_path}")

# ----------------------------------------------------------------------
# 7. 显示（可选）
# ----------------------------------------------------------------------
plt.show()