# 2025-03-24 Hailong Liu
# 用于绘制 .pvtu 文件中的场数据，包括温度场、粘度场等
# 用法：将文件路径替换为实际的路径，将场数据名称替换为实际的场数据名称，运行脚本即可绘制场数据的图像

import pyvista as pv
import numpy as np
import os
# 加载 .pvtu 文件




def plot_T(dataset, T, save_path, number):

    # 创建一个渲染窗口
    plotter = pv.Plotter(window_size=(1000, 200))
    # 添加数据到渲染窗口，手动设置数据范围
    plotter.add_mesh(dataset, scalars=T, clim=[T.min(), T.max()], cmap='coolwarm',show_scalar_bar=False)
    # 添加颜色条
    plotter.add_scalar_bar(
        title="T",            # 颜色条的标题
        vertical = True,      # 垂直放置
        position_x=0.92,       # 颜色条的 x 位置（0 到 1 之间）
        position_y=0.3,       # 颜色条的 y 位置（0 到 1 之间）
        width=0.03,            # 颜色条的宽度（0 到 1 之间）
        height=0.5,          # 颜色条的高度（0 到 1 之间）
        label_font_size=12,   # 标签字体大小
        title_font_size=16    # 标题字体大小
    )

    # 设置视图和相机位置
    plotter.view_xy()
    plotter.camera_position = 'xy'

    # 调整相机的缩放比例
    plotter.camera.zoom(4.2)  # 缩放比例，可以根据需要调整

    plotter.save_graphic(save_path + f'\\{number}' + '_T.svg')
    plotter.save_graphic(save_path + f'\\{number}' + '_T.pdf')
    plotter.screenshot(save_path + f'\\{number}' + '_T.png', transparent_background=True) # 保存为 PNG 格式,带透明度
    # 显示渲染窗口
    # plotter.show()

def plot_viscosity(dataset, viscosity, save_path, number):
    # 创建一个渲染窗口
    plotter = pv.Plotter(window_size=(1000, 200))
    # 添加数据到渲染窗口，手动设置数据范围
    plotter.add_mesh(dataset, scalars=viscosity, clim=[viscosity.min(), viscosity.max()], cmap='coolwarm',show_scalar_bar=False, log_scale=True)
    # 添加颜色条
    plotter.add_scalar_bar(
        title="viscosity",            # 颜色条的标题
        vertical = True,      # 垂直放置
        position_x=0.92,       # 颜色条的 x 位置（0 到 1 之间）
        position_y=0.3,       # 颜色条的 y 位置（0 到 1 之间）
        width=0.03,            # 颜色条的宽度（0 到 1 之间）
        height=0.5,          # 颜色条的高度（0 到 1 之间）
        label_font_size=12,   # 标签字体大小
        title_font_size=16    # 标题字体大小
    )
    # 设置视图和相机位置
    plotter.view_xy()
    plotter.camera_position = 'xy'
    # 调整相机的缩放比例
    plotter.camera.zoom(4.2)  # 缩放比例，可以根据需要调整
    plotter.save_graphic(save_path + f'\\{number}' + r'_viscosity.svg')
    plotter.save_graphic(save_path + f'\\{number}' + r'_viscosity.pdf')
    plotter.screenshot(save_path + f'\\{number}' + r'_viscosity.png', transparent_background=True) # 保存为 PNG 格式,带透明度
    # 显示渲染窗口
    # plotter.show()

def plot_strain_rate(dataset, strain_rate, save_path, number):
    # 创建一个渲染窗口
    plotter = pv.Plotter(window_size=(1000, 200))
    # 添加数据到渲染窗口，手动设置数据范围
    plotter.add_mesh(dataset, scalars=strain_rate, clim=[strain_rate.min(), strain_rate.max()], cmap='coolwarm',show_scalar_bar=False, log_scale=True)
    # 添加颜色条
    plotter.add_scalar_bar(
        title="strain_rate",            # 颜色条的标题
        vertical = True,      # 垂直放置
        position_x=0.92,       # 颜色条的 x 位置（0 到 1 之间）
        position_y=0.3,       # 颜色条的 y 位置（0 到 1 之间）
        width=0.03,            # 颜色条的宽度（0 到 1 之间）
        height=0.5,          # 颜色条的高度（0 到 1 之间）
        label_font_size=12,   # 标签字体大小
        title_font_size=16    # 标题字体大小
    )
    # 设置视图和相机位置
    plotter.view_xy()
    plotter.camera_position = 'xy'
    # 调整相机的缩放比例
    plotter.camera.zoom(4.2)  # 缩放比例，可以根据需要调整
    plotter.save_graphic(save_path + f'\\{number}' + r'_strain_rate.svg')
    plotter.save_graphic(save_path + f'\\{number}' + r'_strain_rate.pdf')
    plotter.screenshot(save_path + f'\\{number}' + r'_strain_rate.png', transparent_background=True) # 保存为 PNG 格式,带透明度
    # 显示渲染窗口
    # plotter.show()

def plot_viscosity_velocity(dataset, viscosity, velocity):
    # 创建一个渲染窗口
    plotter = pv.Plotter(window_size=(1000, 200))

    # 选择要可视化的点（每隔10个点选择一个）
    stride = 50

    # 选取新的 points
    new_points = dataset.points[::stride]

    # 选取新的 velocity 数据
    velocity = dataset['velocity']
    new_velocity = velocity[::stride]

    # 创建新的 UnstructuredGrid 对象
    dataset_stride = pv.UnstructuredGrid()
    dataset_stride.points = new_points
    dataset_stride['velocity'] = new_velocity

    # 创建箭头表示速度场
    arrows = pv.Arrow()
    glyphs = dataset_stride.glyph(orient='velocity', scale=1.0, factor=1.0, geom=arrows)
    plotter.add_mesh(glyphs,label='Velocity',cmap='viridis')

    # 添加数据到渲染窗口，手动设置数据范围
    plotter.add_mesh(dataset, scalars=viscosity, clim=[viscosity.min(), viscosity.max()], cmap='coolwarm',show_scalar_bar=False, log_scale=True)
    # 添加颜色条
    # plotter.add_scalar_bar(
    #     title="viscosity",            # 颜色条的标题
    #     vertical = True,      # 垂直放置
    #     position_x=0.92,       # 颜色条的 x 位置（0 到 1 之间）
    #     position_y=0.3,       # 颜色条的 y 位置（0 到 1 之间）
    #     width=0.03,            # 颜色条的宽度（0 到 1 之间）
    #     height=0.5,          # 颜色条的高度（0 到 1 之间）
    #     label_font_size=12,   # 标签字体大小
    #     title_font_size=16    # 标题字体大小
    # )
    # 设置视图和相机位置
    plotter.view_xy()
    plotter.camera_position = 'xy'
    # plotter.add_legend()
    # 调整相机的缩放比例
    plotter.camera.zoom(4.2)  # 缩放比例，可以根据需要调整
    plotter.save_graphic(file_path + r'\viscosity_velocity.svg')
    plotter.save_graphic(file_path + r'\viscosity_velocity.pdf')
    # 显示渲染窗口
    # plotter.show()


file_path = r'E:\backup\DoubleSubduction\model\double_plstc_subeen\gwb_add\ts\t10\output_t10\solution'

for number in ['00000','00050', '00100', '00150',  '00200']:
    filename = file_path + r'\solution-' + number + '.pvtu'
    save_path = file_path + r'\\plot' 
    dataset = pv.read(filename)

    if os.path.exists(save_path):
        print(f"Directory {save_path} already exists.")
    else:
        os.makedirs(save_path)
        print(f"Directory {save_path} created.")

    # 提取特定的场数据
    point_data = dataset.point_data
    cell_data = dataset.cell_data

    # 打印可用的场数据
    print("Point Data:")
    print(point_data.keys())
    print("Cell Data:")
    print(cell_data.keys())

    # 选择一个场数据进行可视化
    T = point_data['T']  # 替换为实际的场数据名称
    viscosity = point_data['viscosity']
    strain_rate = point_data['strain_rate']  # 替换为实际的场数据名称

    plot_T(dataset, T, save_path, number)
    plot_viscosity(dataset, viscosity, save_path, number)
    plot_strain_rate(dataset, strain_rate, save_path, number)
    # plot_viscosity_velocity(dataset, viscosity, dataset['velocity'])
