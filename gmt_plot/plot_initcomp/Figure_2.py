import os
# from numpy import *           # import numpy modules

# Specify GMT settings
# os.system ('gmtset PS_MEDIA letter; gmtset PS_PAGE_ORIENTATION portrait')
# os.system ('gmtset MAP_FRAME_TYPE fancy; gmtset MAP_FRAME_WIDTH 0.01i')
# os.system ('gmtset MAP_FRAME_PEN 0.75p; gmtset MAP_TICK_LENGTH -0.05i')
# os.system ('gmtset FONT_ANNOT_PRIMARY 7p;gmtset FONT_ANNOT_PRIMARY 0.05i')
# os.system ('gmtset MAP_TITLE_OFFSET 0.05i; gmtset PROJ_LENGTH_UNIT inch')
import subprocess

# 合并所有的 gmtset 命令
gmtset_commands = """
gmt begin Figure_2 ps
gmt set PS_MEDIA letter
gmt set PS_PAGE_ORIENTATION portrait
gmt set MAP_FRAME_TYPE fancy;
gmt set MAP_FRAME_WIDTH 0.01i;
gmt set MAP_FRAME_PEN 0.75p;
gmt set MAP_TICK_LENGTH -0.05i;
gmt set FONT_ANNOT_PRIMARY 7p;
gmt set FONT_ANNOT_PRIMARY 0.05i;
gmt set MAP_TITLE_OFFSET 0.05i;
gmt set PROJ_LENGTH_UNIT inch
"""

try:
    # 执行合并后的命令
    result = subprocess.run(gmtset_commands, shell=True, check=True, capture_output=True, text=True)
    print("命令执行成功，标准输出：")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print("命令执行失败，错误信息：")
    print(e.stderr)

# Plot basemap for compositional fields
os.system ('psbasemap -K -JX4.5i/1.35i -R0/1000/-240/0 -B0sew -X+0.75i -Y+1.0i -P > fig.ps')

# Plot compositional fields
#os.system ('makecpt -Ctopo -T0/10/1 > color.cpt')
comp = "psxy -K -O -JX -R  -A -L -Celmatpr.cpt <<END>> fig.ps \n\
> -Z 1 \n 0   0 \n 1000   0 \n 1000 -20 \n 0 -20 \n 0   0 \n\
> -Z 2 \n 0 -20 \n 1000 -20 \n 1000 -40 \n 0 -40 \n 0 -25 \n\
> -Z 3 \n 0 -40 \n 1000 -40 \n 1000 -120 \n 0 -120 \n 0 -40 \n\
> -Z 4 \n 0 -120 \n 1000 -120 \n 1000 -240 \n 0 -240 \n 0 -120 \n\
> -Z 8 \n 497 -41 \n 503 -41 \n 503 -51 \n 497 -51 \n 497 -41 \n\
END"; os.system (comp)

# Plot basemap for strength profile plot
os.system ('psbasemap -K -O -JX0.75i/-1.35i -R0/800/0/120 -Ba500f100/a30f10nW  -X+5.25i >> fig.ps')

# Calculate strength profile
from Functions_Figure_2 import calcstrength
v=calcstrength('strength.dat')

# Plot strength values
os.system ('psxy strength1.dat -K -O -R -JX -A -W0.5p,black >> fig.ps')
os.system ('psxy strength2.dat -K -O -R -JX -A  -W0.5p,black,- >> fig.ps')

# Add text
gc = "gmt pstext -O -K -JX8.5i/11i -R0/8.5/0/11 -X-6.00i -Y-1.0i -W <<END>> fig.ps \n\
0.20  2.570 10 0 5 CM a)\n\
3.00  2.420  8 0 5 CM Free Surface\n\
3.00  2.240  8 0 5 CM Crust\n\
3.00  1.840  8 0 5 CM Lithospheric Mantle\n\
3.00  1.340  8 0 5 CM Asthenosphere\n\
3.00  2.040  8 0 5 CM Seed\n\
0.93  1.340  8 0 5 CM Q = 0\n\
5.03  1.340  8 0 5 CM Q = 0\n\
0.75  0.900  8 0 5 CM x = 0 km\n\
5.25  0.900  8 0 5 CM x = 1000 km\n\
0.80  2.420  8 0 5 LM z@-S@- = 0 km\n\
0.80  2.150  8 0 5 LM z@-M@- = 40 km\n\
0.80  1.680  8 0 5 LM z@-LAB@- = 120 km\n\
0.80  1.080  8 0 5 LM z@-B@- = 240 km\n\
5.20  2.420  8 0 5 RM T@-S@- = 0 @+o@+C\n\
5.20  2.150  8 0 5 RM T@-M@- = 602 @+o@+C\n\
5.20  1.680  8 0 5 RM T@-LAB@- = 1300 @+o@+C\n\
5.20  1.080  8 0 5 RM T@-B@- = 1360 @+o@+C\n\
0.38  2.470  7 0 5 CM 1\n\
0.38  2.425  7 0 5 CM _\n\
0.38  2.370  7 0 5 CM 2\n\
0.50  2.420  8 0 5 CM V@-ext@-\n\
5.44  2.470  7 0 5 CM 1\n\
5.44  2.425  7 0 5 CM _\n\
5.44  2.370  7 0 5 CM 2\n\
5.65  2.420  8 0 5 RM V@-ext@-\n\
5.55  1.170  8 0 5 RM V@-in@-\n\
3.03  0.870  8 0 5 RM V@-in@-\n\
5.85  2.570 10 0 5 CM b)\n\
6.00  2.430  8 0 5 LM Strength (MPa) \n\
6.48  2.250  7 0 5 CM 500 \n\
6.52  2.070  6 0 5 CM @~f@~ = 20@+o@+ \n\
6.18  1.820  6 0 5 CM @~f@~ = 10@+o@+ \n\
6.05  0.950  7 0 5 CM (km)\n\
END" ; os.system (gc)

# Add line between 120 km depth on composition plot and base of strength plot
com="psxy -K -O -JX -R -W0.25p,red <<END>> fig.ps \n\
5.255  1.670\n\
5.995  0.995\n\
END"; os.system (com)

# Add arrows for sides of composition plot between 0 km and 120 km
com="psxy -K -O -JX -R -SVb0.015i/0.050i/0.050i -Gblack <<END>> fig.ps \n\
5.375  2.294  90.0 0.15\n\
5.375  2.181  90.0 0.15\n\
5.375  2.067  90.0 0.15\n\
5.375  1.956  90.0 0.15\n\
5.375  1.844  90.0 0.15\n\
5.375  1.731  90.0 0.15\n\
0.625  2.294 270.0 0.15\n\
0.625  2.181 270.0 0.15\n\
0.625  2.067 270.0 0.15\n\
0.625  1.956 270.0 0.15\n\
0.625  1.844 270.0 0.15\n\
0.625  1.731 270.0 0.15\n\
END"; os.system (com)

# Add arrows for sides of composition plot between 120 km and 240 km
com="psxy -K -O -JX -R -SVb0.015i/0.025i/0.025i -Gblack <<END>> fig.ps \n\
5.375  1.619  90.0 0.12\n\
5.375  1.506  90.0 0.053\n\
5.375  1.394 270.0 0.00\n\
5.375  1.281 270.0 0.044\n\
5.375  1.169 270.0 0.044\n\
5.375  1.056 270.0 0.044\n\
0.625  1.619 270.0 0.12\n\
0.625  1.506 270.0 0.053\n\
0.625  1.394  90.0 0.00\n\
0.625  1.281  90.0 0.044\n\
0.625  1.169  90.0 0.044\n\
0.625  1.056  90.0 0.044\n\
END"; os.system (com)

# Add arrows for base of composition plot between 100 km and 700 km
com="psxy -K -O -JX -R -SVb0.015i/0.025i/0.025i -Gblack <<END>> fig.ps \n\
1.313  0.92  0.0 0.046\n\
1.594  0.92  0.0 0.046\n\
1.875  0.92  0.0 0.046\n\
2.156  0.92  0.0 0.046\n\
2.438  0.92  0.0 0.046\n\
2.719  0.92  0.0 0.046\n\
3.000  0.92  0.0 0.046\n\
3.281  0.92  0.0 0.046\n\
3.563  0.92  0.0 0.046\n\
3.844  0.92  0.0 0.046\n\
4.125  0.92  0.0 0.046\n\
4.406  0.92  0.0 0.046\n\
4.688  0.92  0.0 0.046\n\
END"; os.system (com)

# Add text
gc = "gmt pstext -JX8.5i/4i -R0/8.5/7/11 <<END>> fig.ps \n\
-1. -1. 8 0 5 CM WillNotAppear \n\
END" ; os.system (gc)

# Add bounding box
# os.system ('ps2epsi fig.ps; psconvert -Au -Tf fig.epsi; mv fig.pdf Figure_2.pdf');
# os.system ('gswin64c -sDEVICE=pdfwrite -sOutputFile=fig.pdf fig.ps\n' \
# ' psconvert -Au -Tf fig.epsi\n' \
# ' mv fig.pdf Figure_2.pdf');
os.system('''gswin64c -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -sOutputFile=fig.pdf fig.ps
          psconvert -Au -Tf fig.epsi
          move fig.pdf Figure_2.pdf''')
os.system('gmt end show')
# os.system ('rm *.eps* *.ps *.dat *.pyc')
