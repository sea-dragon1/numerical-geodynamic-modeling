@echo off
gmt begin ma_ph2 jpg

  REM 1. 地形底图
  gmt grdimage @earth_relief_01m -JM15c -R90/150/0/30 -Baf -BWSen -I+d

  REM 2. 创建临时数据文件并绘制锯齿线
  REM 巽他海沟
  (
    echo 94 5
    echo 98 2.5
    echo 102 0.5
    echo 105 0
  ) > sunda.txt
  gmt psxy sunda.txt -W1p,red -Sf0.3c/0.15c+l+t -R -JM

  REM 菲律宾海沟
  (
    echo 125 5
    echo 127 10
    echo 129 15
    echo 130 20
  ) > philippine.txt
  gmt psxy philippine.txt -W1p,red -Sf0.3c/0.15c+r+t -R -JM

  REM 马尼拉海沟
  (
    echo 118 14
    echo 120 18
    echo 122 22
  ) > manila.txt
  gmt psxy manila.txt -W1p,red -Sf0.3c/0.15c+l+t -R -JM

  REM 爪哇延伸段
  (
    echo 107 0
    echo 110 -2
    echo 113 -4
  ) > java_ext.txt
  gmt psxy java_ext.txt -W1p,red -Sf0.3c/0.15c+l+t -R -JM

  REM 3. 叠加海岸线
  gmt coast -R -JM -W0.5p,black -Df

  REM 清理临时文件（可选）
  del sunda.txt philippine.txt manila.txt java_ext.txt

gmt end show