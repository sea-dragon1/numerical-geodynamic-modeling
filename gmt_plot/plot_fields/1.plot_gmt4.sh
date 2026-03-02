#! /bin/bash
exec &> stdout_plot

main() {

# Model name
name=t4

# Set plot size and properties
R='-R0/2000/-10/700'; J='-JX6.0i/-1.65i';  B='-B250f100/100f50SWe';

# Set general gmt parameters
set_gmt_general_parameters

# Define time steps 
#steps=( 20 40 60 80 100 120 140 160 180 200)
#steps=( 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200)
#steps=( 30 60 90 120 150)
#steps=( 20 40 60 80 100 )
#steps=( 15 30 45 60 75 90)
steps=( 25 )
# Define parameters to plot
params=( viscosity )
# params=( density )
#params=( OceanicMantle )

# Loop through time steps
for step in ${steps[@]}; do

   # Loop through parameters
   for param in ${params[@]}; do 

     # Define postscript file name
     psfile=$name'.'$step'.'$param'.ps'

     # Define data file name
     infile='E:\backup\DoubleSubduction\model\double_plstc_subeen\gwb_add\ts\t4\output_t4\solution\out\'$name'.'$step'.'$param'.out'

     # Define plot position and plot basemap
     P='-X+0.5i -Y+8.40i'; basemap $R $J $B $P -K    >  $psfile

     # Make color scale
     color_scale

     # Plot field
     psxy $infile $R $J $B -A -m -L -H -Ccolor.cpt -K -O  >> $psfile

     # Make color scale for temperature contour plot
     color_scale

     # Plot color bar
     #plot_colorbar

     # Plot figure label
     plot_figure_labels

     # Convert .ps to .pdf file and enforce a tight bounding box (use -P to prevent rotation)
     ps2raster $psfile -A -Tg -E600 -P

     # Remove color file and .ps files
     rm *.cpt *.ps


  done

done

} 

function plot_figure_labels() {

# Read in model time
myr=$(head -n 1 $infile)
myr2=$(printf "%4.1f" $myr) # round value to have only 1 significant digit


echo   5  -25  8  0 5 LM  $myr2 'Myr'| pstext $R $J -N -K -O >> $psfile
#echo   5  -25  8  0 5 LM  $myr2 'Myr'| pstext $R $J -N -Gblack -K -O >> $psfile (make note black)
echo -40  1100  8  0 5 CM '(km)'      | pstext $R $J -N -K -O >> $psfile
echo 250   -5  8   0 5 CM $label     | pstext $R $J -N    -O >> $psfile

}


function plot_colorbar() {

gmt psscale -D2.5i/-0.25i/5.0i/0.1ih $S -K -O -Ccolor.cpt $L >> $psfile
echo 1000 900 9 0 5 CM $unit |pstext $R $J -N -K -O >> $psfile

}


function color_scale() {


if [ "$param" == "pres" ] ; then
  nfb='--COLOR_NAN=- --COLOR_BACKGROUND=- --COLOR_FOREGROUND=-'
  makecpt -T0/4000/100 -Z -M $nfb > color.cpt
  S='--D_FORMAT=%3.0f'; unit='Pressure (MPa)'; L=' -B500'
fi

if [ "$param" == "srrt" ] ; then
  nfb='--COLOR_NAN=- --COLOR_BACKGROUND=- --COLOR_FOREGROUND=-'
  makecpt -T-20/-13/0.1 -Crainbow -Z -M $nfb > color.cpt
  S='--D_FORMAT=%3.0f'; unit='Logarithmic Viscosity (Pa s)'; L=' -B1'
fi

if [ "$param" == "temp" ] ; then
  nfb='--COLOR_NAN=- --COLOR_BACKGROUND=- --COLOR_FOREGROUND=-'
  makecpt -T200/1600/100 -Crainbow -Z -M $nfb > color.cpt
  S='--D_FORMAT=%4.f'; unit='Temperature (K)'; L=' -B200'
fi

if [ "$param" == "viscosity" ] ; then
  nfb='--COLOR_NAN=- --COLOR_BACKGROUND=- --COLOR_FOREGROUND=-'
  makecpt -T20/26/0.1 -Cseis -Z -M $nfb > color.cpt
  S='--D_FORMAT=%3.0f'; unit='Logarithmic Viscosity (Pa s)'; L=' -B1'
fi

if [ "$param" == "OceanicMantle" ] ; then
  echo "1   173  150  122    2  173  150  122"  > color.cpt
  echo "2   197  140  121    3  197  140  121" >> color.cpt
  echo "3   119  196  132    4  119  196  132" >> color.cpt
  echo "4   255  255  255    5  255  255  255" >> color.cpt
  S='--D_FORMAT=%3.0f'; unit='Logarithmic Viscosity (Pa s)'; L=' -B1'
fi


}

function set_gmt_general_parameters() {

PAPER_MEDIA=11x17 PAGE_ORIENTATION=landscape
FRAME_PEN=1p LABEL_FONT_SIZE=14p ANOT_FONT_SIZE=8p
ANNOT_OFFSET_PRIMARY=0.025i TICK_LENGTH=0.05i
N_HEADER_RECS=1

LC_NUMERIC="en_US.UTF-8" # fix for printing issue that came up with $sec variable

}

main "$@"
