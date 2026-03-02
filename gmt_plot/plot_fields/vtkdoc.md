## vtk document

1. 创建读取器
    reader = vtk.vtkXMLPUnstructuredGridReader()
2. 读取file_name文件
    reader.SetFileName(file_name)
    reader.Update()
3. 获取坐标数据
    nodes_vtk_array= reader.GetOutput().GetPoints().GetData()
4. 将vtk数据转为np格式上
    nodes_numpy_array = vtk_to_numpy(nodes_vtk_array)
5. 获取具体场数据
    filed 为 string格式 'viscosity' 'T'...
    field_vtk_array = reader.GetOutput().GetPointData().GetArray(field)
    field_numpy_array = vtk_to_numpy(field_vtk_array)

本文件下，
整体运行逻辑
1. 利用convertData_700.py将pvtu数据转换成out数据；
2. 利用plot_gmt*.sh读取out数据绘制图像；