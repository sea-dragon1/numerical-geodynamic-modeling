@echo off
rem 开始一个新的 GMT 会话，输出为 PDF 格式
gmt begin ma_ph1 pdf

rem 绘制地球地形浮雕图像
gmt grdimage @earth_relief_01m -JM15c -R90/145/0/30 -Baf -BWSen -I+d

rem 绘制板块边界
rem 假设板块边界数据文件为 plate_boundaries.gmt，你需要根据实际情况替换
gmt plot plate_boundaries.gmt -W1p,red

rem 标注缅甸位置
rem 缅甸大致中心经纬度为 95E, 21N，你可以根据需求调整
echo 95 21 Myanmar | gmt text -F+f12p,Helvetica-Bold,black+jCM

rem 标注板块名称
rem 印度板块大致位置，这里只是示例，你可以根据实际情况调整
echo 100 10 Indian Plate | gmt text -F+f12p,Helvetica-Bold,black+jCM

rem 南中国海板块大致位置
echo 115 15 South China Sea Plate | gmt text -F+f12p,Helvetica-Bold,black+jCM

rem 菲律宾海板块大致位置
echo 130 15 Philippine Sea Plate | gmt text -F+f12p,Helvetica-Bold,black+jCM

rem 太平洋板块大致位置
echo 140 20 Pacific Plate | gmt text -F+f12p,Helvetica-Bold,black+jCM

rem 结束 GMT 会话并显示结果
gmt end show

pause