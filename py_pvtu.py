# 2025-03-24 Hailong Liu
# 用于绘制 .pvtu 文件中的场数据，包括温度场、粘度场等
# 用法：将文件路径替换为实际的路径，将场数据名称替换为实际的场数据名称，运行脚本即可绘制场数据的图像
# 用ml环境
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np
import argparse
import os
import re
import gc

# 加载 .pvtu 文件

# 绘制单个标量物理场数据的函数
def plot_field(dataset, field, field_name, save_path, number):

    # 创建一个渲染窗口
    plotter = pv.Plotter(window_size=(1000, 200),off_screen=True)
    # 添加数据到渲染窗口，手动设置数据范围
    plotter.add_mesh(dataset, 
                     scalars=field, 
                     clim=[field.min(), 
                    field.max()], 
                    cmap='coolwarm',
                    show_scalar_bar=False, 
                    log_scale=False if field_name == 'T' else True)
    # 添加颜色条
    plotter.add_scalar_bar(
        title=field_name,            # 颜色条的标题
        vertical = True,      # 垂直放置
        position_x=0.92,       # 颜色条的 x 位置（0 到 1 之间）
        position_y=0.3,       # 颜色条的 y 位置（0 到 1 之间）
        width=0.03,            # 颜色条的宽度（0 到 1 之间）
        height=0.5,          # 颜色条的高度（0 到 1 之间）
        label_font_size=12,   # 标签字体大小
        title_font_size=16    # 标题字体大小
    )
    # 添加文字
    plotter.add_text(number[2:4]+'MA', position='lower_left', font_size=5, color="black")
    # 设置视图和相机位置
    plotter.view_xy()
    plotter.camera_position = 'xy'

    # 调整相机的缩放比例
    plotter.camera.zoom(4.2)  # 缩放比例，可以根据需要调整

    plotter.save_graphic(save_path + f'\\{number}_{field_name}.svg')
    plotter.save_graphic(save_path + f'\\{number}_{field_name}.pdf')
    plotter.screenshot(save_path + f'\\{number}_{field_name}.png', transparent_background=True) # 保存为 PNG 格式,带透明度
    # 显示渲染窗口
    plotter.show(auto_close=True)

# 左下角横向图例（方块色块）
def add_horizontal_legend(plotter, custom_colormap, scalar_fields):
    # 1. 基础图例框（横向放置）
    plotter.add_legend(
        size=(0.4, 0.08),           # 横向宽度增加，高度减小
        bcolor="white", 
        border=True, 
        loc="lower left"            # 左下角位置
    )

    # 2. 准备文本和位置（横向排列）
    legend_items = []
    for key, color in custom_colormap.items():
        example_field = next((f for f in scalar_fields if f.endswith(key)), key)
        legend_items.append(f"{key.upper()}: {example_field}")

    # 3. 文本位置（横向分布）
    text_x_positions = np.linspace(0.12, 0.38, len(custom_colormap))  # 从左到右
    text_y_position = 0.03  # 垂直位置（靠近底部）
    text_positions = np.column_stack([
        text_x_positions,
        np.full(len(custom_colormap), text_y_position),
        np.zeros(len(custom_colormap))
    ])

    plotter.add_point_labels(
        text_positions,
        legend_items,
        font_size=8,
        text_color="black",
        show_points=False
    )

    # 4. 方块色块（使用小立方体）
    cube_size = 0.008  # 方块大小
    cube_x_positions = np.linspace(0.05, 0.35, len(custom_colormap))  # 方块在文本左侧
    cube_positions = np.column_stack([
        cube_x_positions,
        np.full(len(custom_colormap), text_y_position + 0.01),  # 稍微抬高
        np.zeros(len(custom_colormap))
    ])

    # 创建小立方体（方块色块）
    cubes = pv.Cube(cube_positions, cube_size).triangulate()
    plotter.add_mesh(
        cubes,
        color=list(custom_colormap.values()),
        show_edges=False,
        show_scalar_bar=False
    )

def clean_composition_data(dataset, field, method='auto_threshold'):
    """
    清洗组分数据，去除数值扩散导致的异常值
    """
    raw_data = dataset[field]
    
    if method == 'auto_threshold':
        # 方法1：基于数据分布的自动阈值
        data_mean = np.mean(raw_data)
        data_std = np.std(raw_data)
        
        # 对于主要组分，使用较高阈值；对于次要组分，使用较低阈值
        if field.startswith(('luc', 'llc', 'lmc', 'muc', 'mlc', 'mmc')):
            # 主要组分：严格阈值
            lower_threshold = max(0.001, data_mean - 2*data_std)
            upper_threshold = min(1.5, data_mean + 3*data_std)
        else:
            # 次要组分：宽松阈值
            lower_threshold = max(0.005, data_mean - 3*data_std)
            upper_threshold = min(2.0, data_mean + 4*data_std)
            
    elif method == 'physical_bounds':
        # 方法2：基于物理意义的固定阈值
        if field in ['lmc','lom', 'mmc', 'scmc','rom', 'rrmc']:
            lower_threshold, upper_threshold = 0.1, 1.1

        if field in ['luc', 'llc', 'muc', 'mlc']:
            lower_threshold, upper_threshold = 0.1, 1.2
        elif field in ['lsed', 'loc']:
            lower_threshold, upper_threshold = 0.1, 1.5
        elif field in ['rrwk', 'rwk',  'lwk']:
            lower_threshold, upper_threshold = 0.1, 1.2 # 0.05 能画完，有一定的随机性，多跑几次
        else:
            lower_threshold, upper_threshold = 0.05, 1.1 # 2.0
            
    elif method == 'percentile':
        # 方法3：基于百分位数
        lower_threshold = np.percentile(raw_data[raw_data > 0], 5)  # 非零数据的5%分位数
        upper_threshold = np.percentile(raw_data, 95)  # 总体数据的95%分位数
        upper_threshold = min(upper_threshold, 2.0)  # 设置上限
    
    # 应用阈值
    mask = (raw_data >= lower_threshold) & (raw_data <= upper_threshold)
    
    print(f"{field}: 阈值[{lower_threshold:.4f}, {upper_threshold:.4f}], "
          f"有效单元{np.sum(mask)}/{len(raw_data)}")
    
    return mask, lower_threshold, upper_threshold

# 绘制组分场的函数
def plot_composition(dataset, field_name, save_path, number):

    # 创建一个Plotter对象
    plotter = pv.Plotter(window_size=(4800, 1200),off_screen=True) # (1000, 200)

    # 定义标量场名称
    scalar_fields = [ 'lmc',  'loc', 'lom',  'muc', 'mlc', 'mmc',  'sclc', 'scmc', 'roc', 'rom', 'rroc', 'rrmc', 'lwk','luc', 'llc','scuc', 'rsed', 'rrsed', 'rrwk', 'rwk']

    # 定义颜色映射

    def normalize_color(rgb):
        return tuple(c / 255.0 for c in rgb)

    # custom_colormap = {
    #     'uc': normalize_color((133, 197, 94)),
    #     'lc': normalize_color((0, 128, 0)),
    #     'mc': normalize_color((0, 197, 0)),
    #     'ed': normalize_color((255, 255, 0)),
    #     'oc': normalize_color((0, 255, 255)),
    #     'om': normalize_color((0, 197, 189)),
    #     'wk': normalize_color((255, 165, 0))
    # }
    custom_colormap = {
        "uc":  (0.92, 0.92, 0.96),  # 烟灰白     #EBEBF5
        "lc":  (0.83, 0.82, 0.89),  # 淡藕灰     #D4D1E3
        "ed":  (0.73, 0.71, 0.81),  # 灰藕色     #BAB5CF
        "oc":  (0.62, 0.60, 0.72),  # 藕紫灰     #9E99B8
        "mc":  (0.51, 0.49, 0.63),  # 紫灰       #827D9F
        "om":  (0.40, 0.38, 0.53),  # 深紫灰     #666187
        "wk":  (0.96, 0.96, 0.98),  # 高光烟灰   #F5F5FA
        "bg":  (0.90, 0.90, 0.94)
    }
    # 5. 添加半透明基础网格（可选）
    plotter.add_mesh(dataset, color= custom_colormap['bg'], show_scalar_bar=False)   # 深棕色（SaddleBrown） #2E598C

    # 6. 为每个组分添加掩膜可视化
    for i, field in enumerate(scalar_fields):
        # 提取后两位作为键
        
        key = field[-2:]
        
        # 获取对应颜色（默认白色）
        color = custom_colormap.get(key,(0.18, 0.35, 0.55))  # (164, 157, 123)深棕色（SaddleBrown）作为默认颜色,(0.02, 0.15, 0.30)最深青-蓝
        # 特殊处理某些字段的颜色
        if field == 'scuc' :
            color = custom_colormap['ed']
        elif field == 'sclc':
            color = custom_colormap['oc']
        elif field == 'scmc':
            color = custom_colormap['om']
        elif field == 'rrmc':
            color = custom_colormap['om']

        # 1. 创建布尔掩膜
        # mask = dataset[field] > 1e-6  # 假设背景值小于1e-6
        # mask = (dataset[field] >= 0.0001) & (dataset[field] <= 2.0)
                # 使用改进的数据清洗方法
        mask, lower_thresh, upper_thresh = clean_composition_data(
            dataset, field, method='physical_bounds'
        )
        cell_ids = np.where(mask)[0]
        
        if len(cell_ids) == 0:
            print(f"Skipping {field} as it has no cells above threshold.")
            continue   # <-- 跳过空场，关键！
        # 2. 方法A：使用extract_points（推荐）
        masked_data = dataset.extract_points(np.where(mask)[0])
        # masked_data = dataset.extract_cells(np.where(mask)[0])
        
        # 添加可视化
        mesh = plotter.add_mesh(
            masked_data,
            color=color,
            opacity=1.0,
            label=field,
            name=field,  # 重要：为颜色条命名
            show_scalar_bar=True,  # 显示颜色条
        )
    # plotter.enable_anti_aliasing('ssaa') 
    # plotter.render() 
    # 方法：使用最基础的add_legend + 手动颜色块和文本
    # # 1. 先添加图例框（无内容）
    # plotter.add_legend(size=(0.18, 0.3), bcolor="white", border=True, loc="upper right")

    # # 2. 手动添加颜色块和文本（使用add_point_labels作为替代）
    # legend_items = []
    # for key, color in custom_colormap.items():
    #     example_field = next((f for f in scalar_fields if f.endswith(key)), key)
    #     legend_items.append(f"{key.upper()}: {example_field}")

    # # 3. 在图例框内添加文本（位置需要手动调整）
    # text_positions = np.array([
    #     [0.82, 0.85 - i * 0.04, 0.0]  # x, y, z 位置（根据图例框位置调整）
    #     for i in range(len(custom_colormap))
    # ])

    # plotter.add_point_labels(
    #     text_positions,
    #     legend_items,
    #     font_size=10,
    #     text_color="black",
    #     shadow=False,
    #     show_points=False,  # 不显示点，只显示文本
    #     shape=None
    # )

    # # 4. 手动添加颜色指示方块（使用add_mesh with single point）
    # for i, (key, color) in enumerate(custom_colormap.items()):
    #     # 在图例框左侧添加小颜色方块
    #     square_pos = np.array([[0.80, 0.85 - i * 0.04, 0.0]])  # 调整位置
    #     plotter.add_mesh(
    #         pv.PolyData(square_pos),
    #         color=color,
    #         point_size=10,
    #         render_points_as_spheres=True,
    #         show_scalar_bar=False
    #     )

    # plotter.add_text(number[2:4]+'MA', position='lower_left', font_size=5, color="black")
    # 设置视图和相机位置
    plotter.view_xy()
    plotter.camera_position = 'xy'
    # plotter.render()
    # 调整相机的缩放比例
    plotter.camera.zoom(4.2)  # 缩放比例，可以根据需要调整
    plotter.save_graphic(save_path + f'\\{number}_{field_name}.svg',
                            raster=True,
                            painter=True
    )
    plotter.save_graphic(save_path + f'\\{number}_{field_name}.pdf')
    plotter.screenshot(save_path + f'\\{number}_{field_name}.png', transparent_background=True) # 保存为 PNG 格式,带透明度
    # 显示渲染窗口
    plotter.show()
    plotter.close() 

# matplotlib绘制速度场
def resample_by_physical_distance(dataset, spacing=1000.0):
    """
    将数据集按等物理距离重新采样
    
    参数:
    dataset: 原始数据集
    spacing: 采样间距（单位：米），默认1km
    """
    # 1. 获取原始数据的边界
    bounds = dataset.bounds
    x_min, x_max = bounds[0], bounds[1]
    y_min, y_max = bounds[2], bounds[3]
    z_min, z_max = bounds[4], bounds[5]
    
    print(f"原始数据边界: X[{x_min:.0f}, {x_max:.0f}], Y[{y_min:.0f}, {y_max:.0f}]")
    
    # 2. 创建规则网格
    # 计算网格点数：确保间距约为spacing
    x_points = int((x_max - x_min) / spacing) + 1
    y_points = int((y_max - y_min) / spacing) + 1
    
    # 避免网格点太多
    x_points = min(x_points, 25)
    y_points = min(y_points, 200)
    
    # 创建网格坐标
    xi = np.linspace(x_min, x_max, x_points)
    yi = np.linspace(y_min, y_max, y_points)
    zi = np.array([0])  # 假设是2D数据，Z=0
    
    # 创建网格
    xi_grid, yi_grid, zi_grid = np.meshgrid(xi, yi, zi, indexing='ij')
    
    # 展平网格点
    grid_points = np.column_stack([xi_grid.ravel(), yi_grid.ravel(), zi_grid.ravel()])
    
    print(f"创建规则网格: {x_points} × {y_points} = {len(grid_points)} 个点")
    
    # 3. 从原始数据中提取速度和粘度
    original_points = dataset.points
    velocity = dataset['velocity']
    
    # 4. 插值到规则网格
    # 使用最近邻插值（速度方向敏感）或线性插值
    try:
        # 分离速度分量进行插值
        u_interp = griddata(original_points, velocity[:, 0], grid_points, method='linear', fill_value=0)
        v_interp = griddata(original_points, velocity[:, 1], grid_points, method='linear', fill_value=0)
        w_interp = griddata(original_points, velocity[:, 2], grid_points, method='linear', fill_value=0)
        
        # 合并插值后的速度
        velocity_interp = np.column_stack([u_interp, v_interp, w_interp])
        
        print(f"插值完成，有效点: {np.sum(~np.isnan(u_interp))}/{len(u_interp)}")
        
        # 5. 创建新的数据集
        grid_dataset = pv.StructuredGrid()
        grid_dataset.points = grid_points
        grid_dataset.dimensions = [x_points, y_points, 1]
        grid_dataset['velocity'] = velocity_interp
        
        return grid_dataset
        
    except Exception as e:
        print(f"插值失败: {e}")
        # 如果插值失败，返回原始数据集
        return dataset
def plot_with_equal_spacing(dataset, save_path, number, spacing_km=1.0):
    """使用等物理距离间距绘图"""
    
    # 将km转换为m
    spacing_m = spacing_km * 1000
    
    max_points = 400

    # 重采样到等物理距离网格
    resampled_dataset = resample_by_physical_distance(dataset, spacing=spacing_m)
    
    # 现在可以使用等间隔的点
    # points = resampled_dataset.points
    # velocity = resampled_dataset['velocity']
    
    # print(f"重采样后点数: {len(points)}")

    try:
        # 1. 智能采样数据点
        total_points = resampled_dataset.n_points
        print(f"总点数: {total_points:,}")
        
        # 动态计算采样间隔
        if total_points > 500000:
            stride = max(1, total_points // max_points)  # 大数据集，较少点
        else:
            stride = max(1, total_points // 300)
        
        points = resampled_dataset.points[::stride]
        velocity = resampled_dataset['velocity'][::stride]
        
        print(f"采样后点数: {len(points)}")
     # 2. 提取2D数据
        x = points[:, 0]
        y = points[:, 1]
        u = velocity[:, 0]
        v = velocity[:, 1]
        
        # 3. 计算速度幅度
        speed = np.sqrt(u**2 + v**2)
        print(f"速度范围: [{speed.min():.3e}, {speed.max():.3e}]")
        
        # 4. 计算合适的箭头缩放
        # 基于数据范围计算
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        diagonal = np.sqrt(x_range**2 + y_range**2)
        
        # 计算速度的平均值和中位数
        avg_speed = np.mean(speed[speed > 0]) if np.any(speed > 0) else 0.001
        median_speed = np.median(speed[speed > 0]) if np.any(speed > 0) else 0.001
        
        # 自动计算缩放因子
        # 让箭头长度约为对角线长度的 1/40
        if avg_speed > 0:
            scale_factor = diagonal / (avg_speed * 40) /100000
        else:
            scale_factor = 1.0
            
        print(f"数据范围: X[{x.min():.2f}, {x.max():.2f}], Y[{y.min():.2f}, {y.max():.2f}]")
        print(f"对角线长度: {diagonal:.2f}")
        print(f"平均速度: {avg_speed:.3e}")
        print(f"缩放因子: {scale_factor:.3e}")
        
        # 5. 创建图形，设置合适的大小
        fig, ax = plt.subplots(figsize=(16, 12), dpi=300, facecolor='white')
        
        # 6. 绘制箭头 - 使用计算出的缩放因子
        # 关键：使用合适的缩放参数
        quiver = ax.quiver(
            x, y, u, v, speed,
            cmap='viridis',
            scale=1.0 / (scale_factor * 0.1),  # 调整这个值
            scale_units='inches',
            width=0.001,      # 箭头宽度
            headwidth=2,      # 箭头头部宽度
            headlength=2,     # 箭头头部长度
            headaxislength=2.5,  # 头部轴长
            minlength=0.1,    # 最小长度
            pivot='mid',      # 箭头中点
            angles='xy',      # 基于xy平面
            alpha=0.8,        # 透明度
            zorder=2          # 图层顺序
        )
        
        # 使用自定义格式化函数
        def m_to_km(x, pos):
            """将米转换为公里并格式化"""
            return f'{x/1000:.1f}'
        
        # 应用格式化器
        ax.xaxis.set_major_formatter(plt.FuncFormatter(m_to_km))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(m_to_km))
        
        # 7. 设置坐标轴范围（确保包含所有数据）
        # 添加5%的边距
        x_margin = x_range * 0.05
        y_margin = y_range * 0.05
        
        ax.set_xlim((x.min() - x_margin), (x.max() + x_margin))
        ax.set_ylim((y.min() - y_margin), (y.max() + y_margin))
        
        # 8. 设置图形属性
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Length (km)', fontsize=12)
        ax.set_ylabel('Depth (km)', fontsize=12)
        ax.set_title(f'Velocity Field - Time Step {number}', fontsize=14, pad=20)
        
        # 9. 添加网格（可选）
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # 10. 添加颜色条（调整大小和位置）
        # cbar = plt.colorbar(quiver, ax=ax, 
        #                    fraction=0.01,  # 颜色条宽度比例
        #                    pad=0.02,        # 与图的间距
        #                    aspect=15,       # 颜色条长宽比
        #                 #    format='%.1e'
        #                    )   # 科学计数法格式
        
        # cbar.set_label('Velocity Magnitude', fontsize=11)
        # cbar.ax.tick_params(labelsize=9)
        
        # 11. 调整布局
        plt.tight_layout()
        
        # 12. 保存图像
        output_path = os.path.join(save_path, f'{number}_velocity_matplotlib.png')
        plt.savefig(output_path, 
                   dpi=300, 
                   bbox_inches='tight', 
                   facecolor='white',
                   edgecolor='none')
        
        print(f"Matplotlib图像保存到: {output_path}")
        
        # 13. 也保存SVG矢量格式
        svg_path = os.path.join(save_path, f'{number}_velocity_matplotlib.svg')
        plt.savefig(svg_path, 
                   format='svg',
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none')
        print(f"SVG矢量图保存到: {svg_path}")
        
        plt.close()
        
        # 14. 创建简化版本（如果数据太多）
        if len(points) > 1000:
            print("创建简化版本...")
            plot_simplified_version(x, y, u, v, speed, save_path, number)
        
        return True
        
    except Exception as e:
        print(f"Matplotlib绘图失败: {e}")
        import traceback
        traceback.print_exc()
        return False
def plot_matplotlib_velocity_solution(dataset, save_path, number, max_points=500):
    """最终解决方案：使用matplotlib确保质量，修复不完整问题"""

    
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)
    
    print(f"使用Matplotlib绘制第 {number} 个时间步...")
    
    try:
        # 1. 智能采样数据点
        total_points = dataset.n_points
        print(f"总点数: {total_points:,}")
        
        # 动态计算采样间隔
        if total_points > 500000:
            stride = max(1, total_points // max_points)  # 大数据集，较少点
        else:
            stride = max(1, total_points // 300)
        
        points = dataset.points[::stride]
        velocity = dataset['velocity'][::stride]
        
        print(f"采样后点数: {len(points)}")
        
        # 2. 提取2D数据
        x = points[:, 0]
        y = points[:, 1]
        u = velocity[:, 0]
        v = velocity[:, 1]
        
        # 3. 计算速度幅度
        speed = np.sqrt(u**2 + v**2)
        print(f"速度范围: [{speed.min():.3e}, {speed.max():.3e}]")
        
        # 4. 计算合适的箭头缩放
        # 基于数据范围计算
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        diagonal = np.sqrt(x_range**2 + y_range**2)
        
        # 计算速度的平均值和中位数
        avg_speed = np.mean(speed[speed > 0]) if np.any(speed > 0) else 0.001
        median_speed = np.median(speed[speed > 0]) if np.any(speed > 0) else 0.001
        
        # 自动计算缩放因子
        # 让箭头长度约为对角线长度的 1/40
        if avg_speed > 0:
            scale_factor = diagonal / (avg_speed * 40) /100000
        else:
            scale_factor = 1.0
            
        print(f"数据范围: X[{x.min():.2f}, {x.max():.2f}], Y[{y.min():.2f}, {y.max():.2f}]")
        print(f"对角线长度: {diagonal:.2f}")
        print(f"平均速度: {avg_speed:.3e}")
        print(f"缩放因子: {scale_factor:.3e}")
        
        # 5. 创建图形，设置合适的大小
        fig, ax = plt.subplots(figsize=(16, 12), dpi=300, facecolor='white')
        
        # 6. 绘制箭头 - 使用计算出的缩放因子
        # 关键：使用合适的缩放参数
        quiver = ax.quiver(
            x, y, u, v, speed,
            cmap='viridis',
            scale=1.0 / (scale_factor * 0.1),  # 调整这个值
            scale_units='inches',
            width=0.001,      # 箭头宽度
            headwidth=3,      # 箭头头部宽度
            headlength=3,     # 箭头头部长度
            headaxislength=3.5,  # 头部轴长
            minlength=0.1,    # 最小长度
            pivot='mid',      # 箭头中点
            angles='xy',      # 基于xy平面
            alpha=0.8,        # 透明度
            zorder=2          # 图层顺序
        )
        
        # 使用自定义格式化函数
        def m_to_km(x, pos):
            """将米转换为公里并格式化"""
            return f'{x/1000:.1f}'
        
        # 应用格式化器
        ax.xaxis.set_major_formatter(plt.FuncFormatter(m_to_km))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(m_to_km))
        
        # 7. 设置坐标轴范围（确保包含所有数据）
        # 添加5%的边距
        x_margin = x_range * 0.05
        y_margin = y_range * 0.05
        
        ax.set_xlim((x.min() - x_margin), (x.max() + x_margin))
        ax.set_ylim((y.min() - y_margin), (y.max() + y_margin))
        
        # 8. 设置图形属性
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Length (km)', fontsize=12)
        ax.set_ylabel('Depth (km)', fontsize=12)
        ax.set_title(f'Velocity Field - Time Step {number}', fontsize=14, pad=20)
        
        # 9. 添加网格（可选）
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # 10. 添加颜色条（调整大小和位置）
        # cbar = plt.colorbar(quiver, ax=ax, 
        #                    fraction=0.01,  # 颜色条宽度比例
        #                    pad=0.02,        # 与图的间距
        #                    aspect=15,       # 颜色条长宽比
        #                 #    format='%.1e'
        #                    )   # 科学计数法格式
        
        # cbar.set_label('Velocity Magnitude', fontsize=11)
        # cbar.ax.tick_params(labelsize=9)
        
        # 11. 调整布局
        plt.tight_layout()
        
        # 12. 保存图像
        output_path = os.path.join(save_path, f'{number}_velocity_matplotlib.png')
        plt.savefig(output_path, 
                   dpi=300, 
                   bbox_inches='tight', 
                   facecolor='white',
                   edgecolor='none')
        
        print(f"Matplotlib图像保存到: {output_path}")
        
        # 13. 也保存SVG矢量格式
        svg_path = os.path.join(save_path, f'{number}_velocity_matplotlib.svg')
        plt.savefig(svg_path, 
                   format='svg',
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none')
        print(f"SVG矢量图保存到: {svg_path}")
        
        plt.close()
        
        
        return True
        
    except Exception as e:
        print(f"Matplotlib绘图失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# PyVista简化版本绘制速度场
def plot_velocity_pyvista_simple(dataset, save_path, number, max_points=200):
    """简化的PyVista实现，提高清晰度"""
    
    os.makedirs(save_path, exist_ok=True)
    sample_method = 'physical_distance_center'  # 'physical_distance_max', 'physical_distance_center' or 'number_sacle'

    try:
        # 采样数据
        if sample_method == 'number_sacle':
            print("使用固定数量采样方法...")
            total_points = dataset.n_points
            stride = max(1, total_points // max_points) if total_points > max_points else 1
            points = dataset.points[::stride]
            velocity = dataset['velocity'][::stride]
            
        elif sample_method == 'physical_distance_max':
            print("使用物理距离采样方法...")
            
            # 获取所有原始数据点
            all_points = dataset.points
            all_velocity = dataset['velocity']
            
            # 计算数据边界和范围
            x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
            y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            # 计算采样距离 - 根据数据范围自动确定或使用固定值
            if 'sample_distance' in locals():
                # 如果外部传入了采样距离参数
                grid_size = sample_distance
            else:
                # 自动计算采样距离
                # 根据最大点数计算大致的采样距离
                area = x_range * y_range
                avg_distance = np.sqrt(area / max_points)
                grid_size = avg_distance
            
            print(f"采样距离: {grid_size:.4f}")
            
            # 创建采样网格
            x_edges = np.arange(x_min, x_max + grid_size, grid_size)
            y_edges = np.arange(y_min, y_max + grid_size, grid_size)
            
            # 初始化存储列表
            sampled_points = []
            sampled_velocity = []
            
            # 方法1: 网格内速度最大点采样（推荐）
            for i in range(len(x_edges) - 1):
                for j in range(len(y_edges) - 1):
                    # 定义当前网格
                    x_low, x_high = x_edges[i], x_edges[i+1]
                    y_low, y_high = y_edges[j], y_edges[j+1]
                    
                    # 找到网格内的所有点
                    in_grid = (
                        (all_points[:, 0] >= x_low) & (all_points[:, 0] < x_high) &
                        (all_points[:, 1] >= y_low) & (all_points[:, 1] < y_high)
                    )
                    
                    # 如果网格内有数据点
                    if np.any(in_grid):
                        grid_points = all_points[in_grid]
                        grid_velocities = all_velocity[in_grid]
                        
                        # 选择速度最大的点作为代表
                        grid_speeds = np.linalg.norm(grid_velocities, axis=1)
                        max_idx = np.argmax(grid_speeds)
                        
                        sampled_points.append(grid_points[max_idx])
                        sampled_velocity.append(grid_velocities[max_idx])
            
            # 转换为数组
            points = np.array(sampled_points)
            velocity = np.array(sampled_velocity)
            
            print(f"物理距离采样结果: {len(points)}个点")
            
            # 可选：如果没有采样到点，使用原始点
            if len(points) == 0:
                print("警告：物理距离采样未获得点，使用原始点")
                points = all_points
                velocity = all_velocity

        elif sample_method == 'physical_distance_center':
            print("使用物理距离采样方法...")
            
            all_points = dataset.points
            all_velocity = dataset['velocity']
            
            # 计算数据边界和采样距离
            x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
            y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            if 'sample_distance' in locals():
                grid_size = sample_distance
            else:
                area = x_range * y_range
                avg_distance = np.sqrt(area / max_points)
                grid_size = avg_distance
            
            # 创建网格中心点
            x_centers = np.arange(x_min + grid_size/2, x_max, grid_size)
            y_centers = np.arange(y_min + grid_size/2, y_max, grid_size)
            
            # 创建KDTree进行最近邻搜索
            from scipy.spatial import cKDTree
            tree = cKDTree(all_points[:, :2])  # 只使用XY坐标
            
            # 生成所有网格中心
            grid_centers = np.array([(x, y) for x in x_centers for y in y_centers])
            
            # 找到每个中心最近的点的索引
            distances, indices = tree.query(grid_centers, distance_upper_bound=grid_size/2)
            
            # 过滤掉距离过远的点
            valid_mask = distances < grid_size/2
            
            # 获取有效采样点
            valid_indices = indices[valid_mask]
            points = all_points[valid_indices]
            velocity = all_velocity[valid_indices]

        # 创建点云
        if points.shape[1] == 2:
            print("检测到2D数据，转换为3D...")
            points_3d = np.column_stack((points, np.zeros(len(points))))
            velocity_3d = np.column_stack((velocity, np.zeros(len(velocity))))
        else:
            points_3d = points
            velocity_3d = velocity
        
        # 创建PolyData对象
        point_cloud = pv.PolyData(points_3d)
        point_cloud['velocity'] = velocity_3d
        speed = np.linalg.norm(velocity, axis=1)
        # 设置最大速度阈值
        max_speed_allowed = np.percentile(speed, 80)  # 取95%分位数,缩小最大的速度
        speed_clamped = np.minimum(speed, max_speed_allowed)
        point_cloud['speed'] = speed_clamped
        # point_cloud['speed'] = speed
        
        # 计算缩放因子
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        wind_x, wind_y = int(x_range//1000), int(y_range//1000)
        print(f" wind_x, wind_y: {wind_x}, {wind_y}")
        avg_range = (x_range + y_range) / 2
        avg_speed = np.mean(speed[speed > 0]) if np.any(speed > 0) else 0.001
        scale_factor = avg_range / (avg_speed * 30) if avg_speed > 0 else 0.1

        # 创建箭头几何体（更简洁的设计）
        arrow_geom = pv.Arrow(
            tip_length=0.25,
            tip_radius=0.08,
            tip_resolution=12,
            shaft_radius=0.03,
            shaft_resolution=12
        )

        # 创建箭头
        arrows = point_cloud.glyph(
            orient='velocity',
            scale='speed',
            factor=scale_factor*0.5,  # **调整这个值：越大箭头越长**
            geom=arrow_geom,
            # tolerance=0.00001  # 忽略小箭头
        )
        
        # 设置绘图器
        plotter = pv.Plotter(
            window_size=[wind_x, wind_y],
            off_screen=True
        )
        
        # 只添加箭头，使用颜色映射
        plotter.add_mesh(
            arrows,
            # scalars='speed', # 使用颜色映射,速度大小指示箭头颜色
            color='black',
            # cmap='viridis',
            show_scalar_bar=False,
            # scalar_bar_args={
            #     'title': 'Velocity Magnitude',
            #     'title_font_size': 20,
            #     'label_font_size': 16,
            #     'shadow': True
            # },
            smooth_shading=True,
            specular=0.5,
            specular_power=30
        )
        
        # 优化视图
        plotter.view_xy()
        plotter.camera_position = 'xy'
        plotter.set_background('white')
        
        # 隐藏坐标轴
        plotter.hide_axes()
        
        # 保存高分辨率图像
        output_path = os.path.join(save_path, f'{number}_velocity_pyvista.png')
        
        # # 重置相机以确保设置生效
        # plotter.reset_camera()

        # # 轻微放大
        # plotter.camera.zoom(4)
        plotter.camera.tight()
        # plotter.show_bounds(
        #     grid='front',  # 显示边框线
        #     location='outer',  # 边框位置
        #     color='black',  # 边框颜色
        # )
        
        plotter.screenshot(
            output_path,
            window_size=[wind_x, wind_y],
            transparent_background=True
        )
        plotter.close()
        
        print(f"成功保存到: {output_path}")
        return True
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False

# 组分场与速度场联合绘制
# def plot_combined_field(dataset, save_path, number, 
#                        velocity_max_points=200, composition_opacity=0.7):
#     """
#     将速度场叠加在组分场上绘制
    
#     参数:
#     - dataset: PyVista数据集
#     - save_path: 保存路径
#     - number: 图像编号
#     - velocity_max_points: 速度场最大采样点数
#     - composition_opacity: 组分场透明度 (0-1)
#     """
    
#     os.makedirs(save_path, exist_ok=True)
    
#     try:
#         # ============ 第一部分：绘制组分场（底图）============
        
#         # 定义标量场名称和颜色映射（与plot_composition相同）
#         scalar_fields = ['lmc', 'loc', 'lom', 'muc', 'mlc', 'mmc', 'sclc', 
#                          'scmc', 'roc', 'rom', 'rroc', 'rrmc', 'lwk', 'luc', 
#                          'llc', 'scuc', 'rsed', 'rrsed', 'rrwk', 'rwk']
        
#         custom_colormap = {
#             "uc": (0.92, 0.92, 0.96),  # 烟灰白
#             "lc": (0.83, 0.82, 0.89),  # 淡藕灰
#             "ed": (0.73, 0.71, 0.81),  # 灰藕色
#             "oc": (0.62, 0.60, 0.72),  # 藕紫灰
#             "mc": (0.51, 0.49, 0.63),  # 紫灰
#             "om": (0.40, 0.38, 0.53),  # 深紫灰
#             "wk": (0.96, 0.96, 0.98),  # 高光烟灰
#             "bg": (0.90, 0.90, 0.94)   # 背景色
#         }
        
#         # 创建Plotter对象（更大的窗口以容纳两种数据）
#         plotter = pv.Plotter(
#             window_size=[2400, 1800],
#             off_screen=True
#         )
        
#         # 添加半透明基础网格
#         plotter.add_mesh(
#             dataset, 
#             color=custom_colormap['bg'], 
#             opacity=0.3,  # 更透明的背景
#             show_scalar_bar=False
#         )
        
#         # 为每个组分添加掩膜可视化
#         for i, field in enumerate(scalar_fields):
#             # 提取后两位作为键
#             key = field[-2:]
            
#             # 获取对应颜色
#             color = custom_colormap.get(key, (0.18, 0.35, 0.55))
            
#             # 特殊处理某些字段的颜色
#             if field == 'scuc':
#                 color = custom_colormap['ed']
#             elif field == 'sclc':
#                 color = custom_colormap['oc']
#             elif field == 'scmc' or field == 'rrmc':
#                 color = custom_colormap['om']
            
#             # 使用改进的数据清洗方法
#             mask, lower_thresh, upper_thresh = clean_composition_data(
#                 dataset, field, method='physical_bounds'
#             )
            
#             # 提取有效单元
#             cell_ids = np.where(mask)[0]
#             if len(cell_ids) == 0:
#                 continue  # 跳过空场
            
#             # 提取并添加组分场
#             masked_data = dataset.extract_cells(cell_ids)
#             plotter.add_mesh(
#                 masked_data,
#                 color=color,
#                 opacity=composition_opacity,  # 可调节透明度
#                 label=field,
#                 name=field,
#                 show_scalar_bar=False  # 不显示颜色条，避免混乱
#             )
        
#         # ============ 第二部分：绘制速度场（上层）============
        
#         # 使用物理距离采样方法
#         all_points = dataset.points
#         all_velocity = dataset['velocity']
        
#         # 计算数据边界和采样距离
#         x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
#         y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
#         x_range = x_max - x_min
#         y_range = y_max - y_min
        
#         # 计算采样距离
#         area = x_range * y_range
#         avg_distance = np.sqrt(area / velocity_max_points)
#         grid_size = avg_distance
        
#         # 创建网格中心点
#         x_centers = np.arange(x_min + grid_size/2, x_max, grid_size)
#         y_centers = np.arange(y_min + grid_size/2, y_max, grid_size)
        
#         # 创建KDTree进行最近邻搜索
#         from scipy.spatial import cKDTree
#         tree = cKDTree(all_points[:, :2])  # 只使用XY坐标
        
#         # 生成所有网格中心
#         grid_centers = np.array([(x, y) for x in x_centers for y in y_centers])
        
#         # 找到每个中心最近的点的索引
#         distances, indices = tree.query(grid_centers, distance_upper_bound=grid_size/2)
        
#         # 过滤掉距离过远的点
#         valid_mask = distances < grid_size/2
        
#         # 获取有效采样点
#         valid_indices = indices[valid_mask]
#         points = all_points[valid_indices]
#         velocity = all_velocity[valid_indices]
        
#         # 转换为3D点云
#         if points.shape[1] == 2:
#             points_3d = np.column_stack((points, np.zeros(len(points))))
#             velocity_3d = np.column_stack((velocity, np.zeros(len(velocity))))
#         else:
#             points_3d = points
#             velocity_3d = velocity
        
#         # 创建点云和速度场数据
#         point_cloud = pv.PolyData(points_3d)
#         point_cloud['velocity'] = velocity_3d
#         speed = np.linalg.norm(velocity, axis=1)
        
#         # 设置最大速度阈值
#         max_speed_allowed = np.percentile(speed, 80)  # 取80%分位数
#         speed_clamped = np.minimum(speed, max_speed_allowed)
#         point_cloud['speed'] = speed_clamped
        
#         # 计算缩放因子
#         avg_range = (x_range + y_range) / 2
#         avg_speed = np.mean(speed[speed > 0]) if np.any(speed > 0) else 0.001
#         scale_factor = avg_range / (avg_speed * 30) if avg_speed > 0 else 0.1
        
#         # 创建箭头几何体
#         arrow_geom = pv.Arrow(
#             tip_length=0.25,
#             tip_radius=0.08,
#             tip_resolution=12,
#             shaft_radius=0.03,
#             shaft_resolution=12
#         )
        
#         # 创建箭头
#         arrows = point_cloud.glyph(
#             orient='velocity',
#             scale='speed',
#             factor=scale_factor * 0.5,  # 调整箭头长度
#             geom=arrow_geom
#         )
        
#         # 添加黑色箭头到plotter
#         plotter.add_mesh(
#             arrows,
#             color='black',
#             show_scalar_bar=False,
#             smooth_shading=True,
#             specular=0.5,
#             specular_power=30
#         )
        
#         # ============ 第三部分：视图设置和保存 ============
        
#         # 优化视图
#         plotter.view_xy()
#         plotter.camera_position = 'xy'
#         plotter.set_background('white')
        
#         # 隐藏坐标轴
#         plotter.hide_axes()
        
#         # 调整相机使图像紧贴
#         plotter.camera.tight()
#         plotter.camera.zoom(1.02)  # 轻微放大确保无白边
        
#         # 保存图像
#         output_path = os.path.join(save_path, f'{number}_combined_field.png')
#         plotter.screenshot(
#             output_path,
#             window_size=[2400, 1800],
#             transparent_background=False
#         )
        
#         # 可选：添加图例（如果需要）
#         # 由于组分场没有颜色条，可以添加一个简单的文本图例
#         legend_text = f"组分场 (透明度: {composition_opacity})\n速度场 (黑色箭头)"
#         plotter.add_text(
#             legend_text, 
#             position='upper_left', 
#             font_size=10, 
#             color='black',
#             font='arial'
#         )
        
#         plotter.close()
#         print(f"成功保存组合图像到: {output_path}")
#         return True
        
#     except Exception as e:
#         print(f"组合绘制错误: {e}")
#         import traceback
#         traceback.print_exc()
#         return False
def plot_combined_field(dataset, save_path, number, 
                       velocity_max_points=200, composition_opacity=1.0):
    """
    将速度场叠加在组分场上绘制
    
    参数:
    - dataset: PyVista数据集
    - save_path: 保存路径
    - number: 图像编号
    - velocity_max_points: 速度场最大采样点数
    - composition_opacity: 组分场透明度 (0-1)
    """
    
    os.makedirs(save_path, exist_ok=True)
    
    try:
        # ============ 设置全局主题允许空网格 ============
        pv.global_theme.allow_empty_mesh = True
        
        # ============ 第一部分：绘制组分场（底图）============
        
        # 定义标量场名称和颜色映射
        scalar_fields = ['lmc', 'loc', 'lom', 'muc', 'mlc', 'mmc', 'sclc', 
                         'scmc', 'roc', 'rom', 'rroc', 'rrmc', 'lwk', 'luc', 
                         'llc', 'scuc', 'rsed', 'rrsed', 'rrwk', 'rwk']
        
        custom_colormap = {
            "uc": (0.92, 0.92, 0.96),  # 烟灰白
            "lc": (0.83, 0.82, 0.89),  # 淡藕灰
            "ed": (0.73, 0.71, 0.81),  # 灰藕色
            "oc": (0.62, 0.60, 0.72),  # 藕紫灰
            "mc": (0.51, 0.49, 0.63),  # 紫灰
            "om": (0.40, 0.38, 0.53),  # 深紫灰
            "wk": (0.96, 0.96, 0.98),  # 高光烟灰
            "bg": (0.90, 0.90, 0.94)   # 背景色
        }
        
        # 计算窗口大小（根据数据范围）
        points = dataset.points
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        aspect_ratio = x_range / y_range
        
        # 设置窗口大小，保持数据宽高比
        base_height = 1600
        base_width = int(base_height * aspect_ratio)
        window_size = [min(base_width, 3200), min(base_height, 2400)]
        # window_size = [int(x_range)//100, int(y_range)//100] 
        # # 绘制不出来 除以1000太小了数据范围: 4000000.0 x 663709.0, 宽高比: 6.03窗口大小: [40000, 6637]

        print(f"数据范围: {x_range:.1f} x {y_range:.1f}, 宽高比: {aspect_ratio:.2f}")
        print(f"窗口大小: {window_size}")
        
        # 创建Plotter对象
        plotter = pv.Plotter(
            window_size=window_size,
            off_screen=True,
            # lighting=True
        )
        
        # 添加半透明基础网格 - 使用浅色背景
        plotter.add_mesh(
            dataset, 
            color=custom_colormap['bg'], 
            opacity=0.15,  # 非常透明的背景，只作为参考
            show_scalar_bar=False,
            style='surface',
            smooth_shading=True
        )
        
        # 为每个组分添加掩膜可视化
        added_fields = []
        for i, field in enumerate(scalar_fields):
            # 提取后两位作为键
            key = field[-2:]
            
            # 获取对应颜色
            color = custom_colormap.get(key, (0.18, 0.35, 0.55))
            
            # 特殊处理某些字段的颜色
            if field == 'scuc':
                color = custom_colormap['ed']
            elif field == 'sclc':
                color = custom_colormap['oc']
            elif field == 'scmc' or field == 'rrmc':
                color = custom_colormap['om']
            
            # 使用改进的数据清洗方法
            mask, lower_thresh, upper_thresh = clean_composition_data(
                dataset, field, method='physical_bounds'
            )
            
            # 提取有效单元 - 使用extract_points而不是extract_cells
            point_indices = np.where(mask)[0]
            if len(point_indices) == 0:
                print(f"跳过 {field}: 没有有效单元 (阈值: [{lower_thresh:.4f}, {upper_thresh:.4f}])")
                continue  # 跳过空场
            
            # 提取并添加组分场
            try:
                # 使用extract_points提取点数据
                masked_data = dataset.extract_points(point_indices)
                
                # 检查提取的网格是否为空
                if masked_data.n_points == 0:
                    print(f"警告: {field} 提取的网格为空，跳过绘制")
                    continue
                
                # 添加组分场到图表 - 使用点模式显示
                plotter.add_mesh(
                    masked_data,
                    color=color,
                    opacity=composition_opacity,
                    label=field,
                    name=field,
                    show_scalar_bar=False,
                    point_size=3.0,  # 控制点的大小
                    render_points_as_spheres=True,  # 点显示为球体
                    style='points'  # 使用点模式
                )
                
                added_fields.append(field)
                print(f"成功添加 {field}: {len(point_indices)} 个有效点")
                
            except Exception as e:
                print(f"处理 {field} 时出错: {e}")
                continue
        
        print(f"总共添加了 {len(added_fields)} 个组分场: {added_fields}")
        
        # ============ 第二部分：绘制速度场（上层）============
        
        # 检查速度场数据是否存在
        if 'velocity' not in dataset.array_names:
            print("警告: 数据集不包含速度场")
        else:
            try:
                print("开始绘制速度场...")
                
                # 获取所有点数据
                all_points = dataset.points
                all_velocity = dataset['velocity']
                
                # 计算采样距离
                x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
                y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
                x_range = x_max - x_min
                y_range = y_max - y_min
                
                # 计算采样距离 - 确保有足够的采样点
                area = x_range * y_range
                desired_points = min(velocity_max_points, int(area / 100))  # 根据面积调整
                avg_distance = np.sqrt(area / desired_points)
                grid_size = max(avg_distance, x_range/100, y_range/100)  # 确保网格不会太小
                
                print(f"速度场采样距离: {grid_size:.1f}, 期望点数: {desired_points}")
                
                # 简单采样方法：等间距采样
                total_points = dataset.n_points
                sample_step = max(1, total_points // desired_points)
                
                # 确保采样步长不会太大
                sample_step = min(sample_step, total_points // 50)
                
                # 采样点
                points = all_points[::sample_step]
                velocity = all_velocity[::sample_step]
                
                # 如果采样点太少，尝试更密集的采样
                if len(points) < 10:
                    sample_step = max(1, total_points // 100)
                    points = all_points[::sample_step]
                    velocity = all_velocity[::sample_step]
                
                print(f"实际采样点数: {len(points)}")
                
                # 转换为3D点云
                if points.shape[1] == 2:
                    points_3d = np.column_stack((points, np.zeros(len(points))))
                    velocity_3d = np.column_stack((velocity, np.zeros(len(velocity))))
                else:
                    points_3d = points
                    velocity_3d = velocity
                
                # 创建点云和速度场数据
                point_cloud = pv.PolyData(points_3d)
                point_cloud['velocity'] = velocity_3d
                speed = np.linalg.norm(velocity, axis=1)
                
                # 设置速度阈值 - 避免极低速箭头太大问题
                min_speed = np.percentile(speed[speed > 0], 20) if np.any(speed > 0) else 0.001
                max_speed = np.percentile(speed, 95)  # 取95%分位数
                
                # 限制速度范围
                speed_clamped = np.clip(speed, min_speed, max_speed)
                
                # 归一化速度
                if max_speed > min_speed:
                    speed_normalized = (speed_clamped - min_speed) / (max_speed - min_speed)
                    speed_scaled = 0.1 + 0.9 * speed_normalized  # 范围0.1-1.0
                else:
                    speed_scaled = np.ones_like(speed_clamped) * 0.5
                
                point_cloud['speed'] = speed_scaled
                
                # 计算缩放因子 - 使箭头大小适中
                avg_range = (x_range + y_range) / 2
                avg_speed = np.mean(speed_scaled)
                scale_factor = avg_range / (avg_speed * 25) if avg_speed > 0 else 0.1
                
                print(f"速度范围: {min_speed:.3f} - {max_speed:.3f}, 缩放因子: {scale_factor:.3f}")
                
                # 创建更简洁的箭头几何体
                arrow_geom = pv.Arrow(
                    tip_length=0.2,
                    tip_radius=0.06,
                    tip_resolution=10,
                    shaft_radius=0.025,
                    shaft_resolution=10
                )
                
                # 创建箭头
                arrows = point_cloud.glyph(
                    orient='velocity',
                    scale='speed',
                    factor=scale_factor * 0.7,  # 调整箭头长度
                    geom=arrow_geom,
                    tolerance=0.0001  # 忽略非常小的箭头
                )
                
                # 检查箭头网格是否为空
                if arrows.n_points > 0:
                    # 添加黑色箭头到plotter
                    plotter.add_mesh(
                        arrows,
                        color='black',
                        show_scalar_bar=False,
                        smooth_shading=True,
                        specular=0.3,
                        specular_power=15,
                        ambient=0.5,
                        diffuse=0.8
                    )
                    print(f"成功添加速度场箭头: {arrows.n_points} 个箭头")
                else:
                    print("警告: 速度场箭头网格为空，尝试更宽松的容差")
                    # 尝试不使用容差
                    arrows = point_cloud.glyph(
                        orient='velocity',
                        scale='speed',
                        factor=scale_factor * 0.7,
                        geom=arrow_geom
                    )
                    if arrows.n_points > 0:
                        plotter.add_mesh(
                            arrows,
                            color='black',
                            show_scalar_bar=False,
                            smooth_shading=True
                        )
                        print(f"成功添加速度场箭头: {arrows.n_points} 个箭头")
                    
            except Exception as e:
                print(f"速度场绘制出错: {e}")
                import traceback
                traceback.print_exc()
        
        # ============ 第三部分：视图设置和保存 ============
        
        print("调整视图设置...")
        
        # 设置视图为XY平面
        plotter.view_xy()
        
        # 设置相机位置 - 正上方看XY平面
        plotter.camera_position = [
            [(x_min + x_max)/2, (y_min + y_max)/2, max(x_range, y_range)],  # 相机位置
            [(x_min + x_max)/2, (y_min + y_max)/2, 0],  # 焦点位置
            [0, 1, 0]  # 上方向
        ]
        
        # 设置正交投影
        plotter.camera.parallel_projection = True

        # 添加完所有 mesh 后，直接 tight
        # plotter.reset_camera()      # 先重置
        # plotter.camera.tight()      # 自动计算最佳 parallel_scale

        # 不要再手动设置 parallel_scale 或多次 reset
        # 隐藏坐标轴
        plotter.hide_axes()
        
        # 设置背景为白色
        plotter.set_background('white')
        
        # 添加时间标签
        plotter.add_text(
            f"Time: {number}",
            position='upper_left',
            font_size=14,
            color='black',
            font='arial'
        )
        
        # 添加图例说明
        legend_text = f"组分场 (透明度: {composition_opacity})\n速度场 (黑色箭头)"
        plotter.add_text(
            legend_text, 
            position='upper_right', 
            font_size=10, 
            color='black',
            font='arial'
        )
        
        # 保存图像
        output_path = os.path.join(save_path, f'{number}_combined_field.png')
        print(f"保存图像到: {output_path}")
        
        # 截图 - 使用透明背景
        plotter.screenshot(
            output_path,
            # window_size=window_size,
            transparent_background=False  # 使用白色背景
        )
        
        # 保存SVG格式（可选）
        try:
            svg_path = os.path.join(save_path, f'{number}_combined_field.svg')
            plotter.save_graphic(svg_path, raster=True, painter=True)
            print(f"同时保存SVG到: {svg_path}")
        except Exception as e:
            print(f"SVG保存失败: {e}")
        
        # 关闭plotter
        plotter.close()
        print(f"成功保存组合图像到: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"组合绘制错误: {e}")
        import traceback
        traceback.print_exc()
        return False
            
# 主函数，处理多个时间步和字段
def plt_pvtu(file_name, time_steps, field_names, args):
    file_path = file_name.format(model=args.model, velocity=args.velocity)

    # file_path = r'E:\backup\DoubleSubduction\model\double_plstc_subeen\gwb_add\ts\t9\v8v8\output_t9\solution'

    use_all_steps = True
    if use_all_steps == True:
        time_steps_all = []  # 用于存储时间步的列表

        # 获取路径下的所有文件和文件夹
        all_files = os.listdir(file_path)
        # 筛选出文件名末尾为 .pvtu 的文件
        pvtu_files = [os.path.join(file_path, f) for f in all_files if f.endswith('.pvtu')]
        # 按时间步排序
        pvtu_files.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
        # 遍历文件列表，提取文件名中的数字部分
        for file in all_files:
            match = re.search(r'solution-(\d+)\.pvtu', file)
            if match:
                time_steps_all.append(match.group(1))

        # print("找到的时间步为:", )

        time_steps = []
        # 每 50 个元素取一个
        for i in range(0, len(time_steps_all)-1, 50):
            time_steps.append(time_steps_all[i])
        # 如果最后不足 50 个，取最后一个
        if len(time_steps_all) % 50 != 0:
            time_steps.append(time_steps_all[-1])
        
        print(f"time steps selected: {time_steps}")

    save_path = file_path + r'\\plot' 
    if os.path.exists(save_path):
        print(f"Directory {save_path} already exists.")
    else:
        os.makedirs(save_path)
        print(f"Directory {save_path} created.")

    for number in time_steps:

        filename = file_path + r'\solution-' + number + '.pvtu'

        dataset = pv.read(filename)

        # 提取特定的场数据
        point_data = dataset.point_data
        cell_data = dataset.cell_data

        # print(dataset)
        # 打印可用的场数据
        # print("Point Data:")
        # print(point_data.keys())
        # print("Cell Data:")
        # print(cell_data.keys())


        for field_name in field_names:
            if field_name == 'composition':
                plot_composition(dataset, field_name, save_path, number)
            elif field_name == 'velocity':
                # plot_viscosity_velocity(dataset, viscosity=point_data["viscosity"], velocity=point_data["velocity"], field_name=field_name, save_path=save_path, number=number)
                # plot_with_equal_spacing(dataset, save_path, number)
                # plot_velocity_pyvista_simple(dataset, save_path, number)  
                # plot_matplotlib_velocity_solution(dataset, save_path, number, max_points=500)
                plot_combined_field(dataset, save_path, number, velocity_max_points=200, composition_opacity=0.6)
            else:
                field = point_data[field_name]
                plot_field(dataset, field, field_name, save_path, number)
        print(f"Finished processing time step {number}")

def main():
    # 定义命令行参数解析器
    parser = argparse.ArgumentParser(description="动态修改文件路径中的占位符")
    parser.add_argument("--model", type=str, default="t9", help="模型占位符的值")
    parser.add_argument("--velocity", type=str, default="v8v8", help="速度占位符的值")

    # 解析命令行参数
    args = parser.parse_args()
    # 定义原始路径模板
    # file_path_template = r'E:\backup\DoubleSubduction\model\double_plstc_subeen\gwb_add\ts\thickness_cftold\t50\t80t50\output_t80t50_cftold\solution' # r'E:\backup\DoubleSubduction\model\double_plstc_subeen\gwb_add\ts\{model}\{velocity}\output_{model}_v\solution'
    # 使用格式化替换占位符
    # paths = [rf'E:\backup\DoubleSubduction\model\double_plstc_subeen\gwb_add\ts\thickness_cftold\t70\t{i}0t70\output_t{i}0t70_cftold' for i in [6,7,8] ]

    # path_suffix = [ 'v4v0', 'v4v4', 'v4v8', 'v8v4', 'v8v8']
    # paths = [rf'E:\backup\DoubleSubduction\model\double_plstc_subeen\gwb_add\ts\diff_v_t70t60\{suffix}\output_t70t60_{suffix}' for suffix in path_suffix]
    paths = [rf'E:\backup\DoubleSubduction\model\double_plstc_subeen\gwb_add\ts\thickness_cftold\t70\t80t70\output_t80t70_cftold']
    time_steps = ['00000','00050', '00100', '00150', '00180',  '00200', '00230','00249']
    # time_steps = ['00137', '00138', '00139', '00140']
    field_names = ['T', 'viscosity', 'strain_rate'] # 'composition',  ['T', 'viscosity', 'strain_rate']  # T, viscosity, strain_rate等场数据名称
    # field_names = ['composition']
    field_names = ['velocity']

    for path in paths:
        plt_pvtu(file_name=path + r'\solution', time_steps=time_steps, field_names=field_names, args=args)
        print(f"Finished processing {path}")

if __name__ == "__main__":
    main()