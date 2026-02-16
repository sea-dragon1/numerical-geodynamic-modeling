import pyvista as pv
from pyvista import PVDReader


import xml.etree.ElementTree as ET
import pyvista as pv
import os

# 加载.pvd文件
def load_pvd(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    collection = root.find('Collection')
    files = []
    for dataset in collection:
        file_path = dataset.attrib['file']
        time = float(dataset.attrib['timestep'])
        files.append((time, file_path))
    files.sort(key=lambda x: x[0])  # 按时间排序
    return files

# 加载.pvd文件并获取时间步和文件路径


def plt_gif(path):
    pvd_file = path + r'\solution.pvd'
    files = load_pvd(pvd_file)

    field_name = 'strain_rate' # T, viscosity, strain_rate等场数据名称

    # 创建一个Plotter对象并设置为off_screen模式
    plotter = pv.Plotter(window_size=(1000, 200),off_screen=True)
    save_path = path + '\\plot' 
    if os.path.exists(save_path):
        print(f"Directory {save_path} already exists.")
    else:
        os.makedirs(save_path)
        print(f"Directory {save_path} created.")

    print(len(files))
    # # 打开GIF文件
    plotter.open_gif(os.path.join(save_path, field_name+'.gif'))


    # 按时间顺序加载.vtu文件并写入GIF帧
    for i in range(0, len(files), 10):
        time, vtu_file = files[i]  # 获取第 i 个元素
        # 读取当前时间步的.vtu文件
        dataset = pv.read(os.path.join(path, vtu_file))
        
        # 清除之前的网格数据
        plotter.clear()

            # 提取特定的场数据
        point_data = dataset.point_data
        cell_data = dataset.cell_data

        # 选择一个场数据进行可视化
        field = point_data[field_name]  # 替换为实际的场数据名称
        
        # 添加当前时间步的网格数据到Plotter
        plotter.add_mesh(dataset, 
                        scalars=field, 
                        clim=[field.min(), field.max()], 
                        cmap='coolwarm',
                        show_scalar_bar=False,
                        log_scale=False if field_name == 'T' else True)
        # 添加文字
        plotter.add_text(str(time//1000000)+'MA', position='lower_left', font_size=5, color="black")
        # 设置视图和相机位置
        plotter.view_xy()
        plotter.camera_position = 'xy'
        # 调整相机的缩放比例
        plotter.camera.zoom(4.2)  # 缩放比例，可以根据需要调整
        # 写入当前帧
        plotter.write_frame()

    # 关闭Plotter并完成GIF文件的保存
    plotter.close()


path = r'E:\backup\DoubleSubduction\model\double_plstc_subeen\gwb_add\ts\thickness_cftold\t70\t60t70\output_t60t70_cftold'
# v0v8 E:\backup\DoubleSubduction\model\double_plstc_subeen\gwb_add\ts\diff_v_t70t60\v0v8\output_t70t60_v0v8_GMRES200
# paths = [rf'E:\backup\DoubleSubduction\model\double_plstc_subeen\gwb_add\ts\thickness_cftold\t50\t{i}0t50\output_t{i}0t50_cftold' for i in [7] ]
paths = [rf'E:\backup\DoubleSubduction\model\double_plstc_subeen\gwb_add\ts\thickness_cftold\r80\l70\output_l70r80']
# path_suffix = [  'v8v0']
# paths = [rf'E:\backup\DoubleSubduction\model\double_plstc_subeen\gwb_add\ts\diff_v_t70t60\{suffix}\output_t70t60_{suffix}' for suffix in path_suffix]
    
for path in paths:
    plt_gif(path)
    print(f"Finished processing {path}")