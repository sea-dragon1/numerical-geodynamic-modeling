gmt begin Figure_2 pdf
gmt set PS_MEDIA letter
gmt set PS_PAGE_ORIENTATION portrait
gmt set MAP_FRAME_TYPE fancy
gmt set MAP_FRAME_WIDTH 0.01i
gmt set MAP_FRAME_PEN 0.75p
gmt set MAP_TICK_LENGTH -0.05i
gmt set FONT_ANNOT_PRIMARY 7p
gmt set FONT_ANNOT_PRIMARY 0.05i;
gmt set MAP_TITLE_OFFSET 0.05i
gmt set PROJ_LENGTH_UNIT inch

# gmt plot gmt basemap for compositional fields
gmt basemap  -JX4.5i/1.35i -R0/1000/-240/0 -Bxaf -Byaf -BWSen -X+0.75i -Y+1.0i

# Plot compositional fields
gmt makecpt -Ctopo -H -T0/10/1 -Z > color.cpt

# 绘制多边形
gmt plot -A -L -Ccolor.cpt << EOF
> -Z1
0   0
1000   0
1000 -20
0 -20

> -Z2
0 -20
1000 -20
1000 -40
0 -40

> -Z3
0 -40
1000 -40
1000 -120
0 -120

> -Z4
0 -120
1000 -120
1000 -240
0 -240

> -Z8
497 -41
503 -41
503 -51
497 -51

EOF

# Plot basemap for strength profile plot
gmt basemap -JX0.75i/-1.35i -R0/800/0/120 -Bxaf -Byaf -BWSen  -X+5.25i

# python 
# Plot strength values
gmt plot strength1.dat -W0.5p,black
gmt plot strength2.dat -W0.5p,black

# Add text labels to the composition plot
gmt text -JX8.5i/11i -R0/8.5/0/11 -X-6.00i -F+f+a+j -Y-1.0i << EOF
0.20  2.570 10 0 MC a)
3.00  2.420  8 0 MC Free Surface
3.00  2.240  8 0 MC Crust
3.00  1.840  8 0 MC Lithospheric Mantle
3.00  1.340  8 0 MC Asthenosphere
3.00  2.040  8 0 MC Seed
0.93  1.340  8 0 MC Q = 0
5.03  1.340  8 0 MC Q = 0
0.75  0.900  8 0 MC x = 0 km
5.25  0.900  8 0 MC x = 1000 km
0.80  2.420  8 0 ML z@-S@- = 0 km
0.80  2.150  8 0 ML z@-M@- = 40 km
0.80  1.680  8 0 ML z@-LAB@- = 120 km
0.80  1.080  8 0 ML z@-B@- = 240 km
5.20  2.420  8 0 MR T@-S@- = 0 @+o@+C
5.20  2.150  8 0 MR T@-M@- = 602 @+o@+C
5.20  1.680  8 0 MR T@-LAB@- = 1300 @+o@+C
5.20  1.080  8 0 MR T@-B@- = 1360 @+o@+C
0.38  2.470  7 0 MC 1
0.38  2.425  7 0 MC _
0.38  2.370  7 0 MC 2
0.50  2.420  8 0 MC V@-ext@-
5.44  2.470  7 0 MC 1
5.44  2.425  7 0 MC _
5.44  2.370  7 0 MC 2
5.65  2.420  8 0 MR V@-ext@-
5.55  1.170  8 0 MR V@-in@-
3.03  0.870  8 0 MR V@-in@-
5.85  2.570 10 0 MC b)
6.00  2.430  8 0 ML Strength (MPa) 
6.48  2.250  7 0 MC 500 
6.52  2.070  6 0 MC @~f@~ = 20@+o@+ 
6.18  1.820  6 0 MC @~f@~ = 10@+o@+ 
6.05  0.950  7 0 MC (km)
END
EOF

# Add line between 120 km depth on composition plot and base of strength plot
gmt plot -W0.25p,red << EOF
5.255  1.670
5.995  0.995
END
EOF

# Add arrows for sides of composition plot between 0 km and 120 km
gmt plot -SVb0.015i/0.050i/0.050i -Gblack << EOF
5.375  2.294  90.0 0.15 
5.375  2.181  90.0 0.15 
5.375  2.067  90.0 0.15 
5.375  1.956  90.0 0.15 
5.375  1.844  90.0 0.15 
5.375  1.731  90.0 0.15 
0.625  2.294 270.0 0.15 
0.625  2.181 270.0 0.15 
0.625  2.067 270.0 0.15 
0.625  1.956 270.0 0.15 
0.625  1.844 270.0 0.15 
0.625  1.731 270.0 0.15 
END
EOF

# Add arrows for sides of composition plot between 120 km and 240 km
gmt plot -SVb0.015i/0.025i/0.025i -Gblack << EOF
5.375  1.619  90.0 0.12 
5.375  1.506  90.0 0.053 
5.375  1.394 270.0 0.00 
5.375  1.281 270.0 0.044 
5.375  1.169 270.0 0.044 
5.375  1.056 270.0 0.044 
0.625  1.619 270.0 0.12 
0.625  1.506 270.0 0.053 
0.625  1.394  90.0 0.00 
0.625  1.281  90.0 0.044 
0.625  1.169  90.0 0.044 
0.625  1.056  90.0 0.044 
END
EOF

# Add arrows for base of composition plot between 100 km and 700 km
gmt plot -SVb0.015i/0.025i/0.025i -Gblack << EOF
1.313  0.92  0.0 0.046 
1.594  0.92  0.0 0.046 
1.875  0.92  0.0 0.046 
2.156  0.92  0.0 0.046 
2.438  0.92  0.0 0.046 
2.719  0.92  0.0 0.046 
3.000  0.92  0.0 0.046 
3.281  0.92  0.0 0.046 
3.563  0.92  0.0 0.046 
3.844  0.92  0.0 0.046 
4.125  0.92  0.0 0.046 
4.406  0.92  0.0 0.046 
4.688  0.92  0.0 0.046 
END
EOF

gmt text -JX8.5i/4i -R0/8.5/7/11 << EOF
-1. -1. 8 0 5 CM WillNotAppear
END
EOF

gmt end show

# rm *.cpt