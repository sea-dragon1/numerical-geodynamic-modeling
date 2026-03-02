import pygmt

# 创建图形
fig = pygmt.Figure()
fig.basemap(region=[90, 150, 0, 30], projection='M15c', frame=True)
fig.grdimage('@earth_relief_01m', shading=True)

# ========== 绘制俯冲带 ==========
# 印度洋板块和欧亚板块的俯冲带
fig.plot(
    data=[[91, 10], [93, 17], [92, 20]],
    pen='1p,red',
    style='f0.3c/0.15c+r+t',
    fill='red',
    label='India-Eurasia Subduction Zone'
)

# 马尼拉海沟
fig.plot(
    data=[[118, 14], [120, 18], [120, 22]],
    pen='1p,red',
    style='f0.3c/0.15c+r+t',   # +r 表示锯齿在右侧
    fill='red',
    label='Manila Trench'
)

# 菲律宾海沟
fig.plot(
    data=[[128, 5], [127, 10], [125, 15]],
    pen='1p,red',
    style='f0.3c/0.15c+l+t',   # +l 表示锯齿在左侧
    fill='red',
    label='Philippine Trench'
)

# 马里亚纳海沟
fig.plot(
    data=[[146, 23], [148, 19], [147, 13], [142, 10]],
    pen='1p,red',
    style='f0.3c/0.15c+r+t',   # +l 表示锯齿在左侧
    fill='red',
    label='Mariana Trench'
)

# ========== 添加板块名称文字 ==========
# 印度板块
fig.text(
    x=85, y=12,
    text='Indian Plate',
    font='10p,Helvetica,black',
    justify='CM'   # 居中对齐
)

# 欧亚板块
fig.text(
    x=110, y=28,
    text='Eurasian Plate',
    font='10p,Helvetica,black',
    justify='CM'
)

# 菲律宾海板块
fig.text(
    x=135, y=20,
    text='Philippine Sea Plate',
    font='10p,Helvetica,black',
    justify='CM'
)

# 太平洋板块
fig.text(
    x=144, y=8,
    text='Pacific Plate',
    font='10p,Helvetica,black',
    justify='CM'
)

# 巽他板块（或东南亚陆块）
fig.text(
    x=105, y=5,
    text='Sunda Plate',
    font='10p,Helvetica,black',
    justify='CM'
)

# 南海板块（或南海陆块）
fig.text(
    x=114, y=15,
    text='South China\n Sea Plate',
    font='10p,Helvetica,black',
    justify='CM'
)

# ========== 手动图例（右下角） ==========
rect_x, rect_y = 140, 0       # 矩形左下角坐标
rect_w, rect_h = 12, 1        # 矩形宽度和高度

# 绘制白色填充、黑色边框的矩形
fig.plot(
    data=[[rect_x, rect_y],
          [rect_x+rect_w, rect_y],
          [rect_x+rect_w, rect_y+rect_h],
          [rect_x, rect_y+rect_h]],
    pen='0.5p,black',
    fill='white',
    close=True
)

# 在矩形内左侧绘制一小段红色锯齿线作为示例
fig.plot(
    data=[[rect_x+0.5, rect_y+0.5], [rect_x+2.5, rect_y+0.5]],
    pen='0.5p,red',
    style='f0.2c/0.1c+l+t',
    fill='red'
)

# 在锯齿线右侧添加文字
fig.text(
    x=rect_x+3.0,
    y=rect_y+0.5,
    text='Subduction zone',
    font='5p,Helvetica,black',
    justify='ML'
)

# ========== 海岸线 ==========
fig.coast(shorelines='0.5p,black')

# 显示和保存
fig.show()
fig.savefig('subduction_zones.png')