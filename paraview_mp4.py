import paraview.simple as pv

# 加载数据
file_path = 'path/to/your/data.pvtu'
reader = pv.OpenDataFile(file_path)

# 显示温度场 T
display = pv.Show(reader)
display.ColorArrayName = ['POINTS', 'T']  # 假设温度场名为 'T'
display.LookupTable = pv.GetColorTransferFunction('T')

# 添加颜色条
pv.ColorLegendRepresentation = pv.ColorLegendRepresentation
pv.ColorLegendRepresentation.LookupTable = pv.GetColorTransferFunction('T')
pv.ColorLegendRepresentation.Title = 'Temperature (T)'
pv.ColorLegendRepresentation.LabelFontSize = 12
pv.ColorLegendRepresentation.TitleFontSize = 14
pv.ColorLegendRepresentation.Position = [0.8, 0.01]
pv.ColorLegendRepresentation.Position2 = [0.1, 0.8]

# 添加时间注释
time_annotation = pv.AnnotateTimeFilter(reader)
time_annotation.Format = "Time: %.2f"
time_display = pv.Show(time_annotation)
time_display.WindowLocation = 'UpperCenter'
time_display.FontSize = 18

# 设置视图
view = pv.GetActiveView()
view.ViewSize = [800, 600]
view.CameraPosition = [0, 0, 100]  # 调整相机位置
view.CameraFocalPoint = [0, 0, 0]  # 调整相机焦点

# 保存动画
animation_scene = pv.GetAnimationScene()
animation_scene.NumberOfFrames = 100  # 设置动画帧数
animation_scene.PlayMode = 'Sequence'
animation_scene.StartTime = 0
animation_scene.EndTime = 10  # 设置时间范围
animation_scene.AnimationTime = 0

pv.SaveAnimation('animation.mp4', view, FrameRate=30)