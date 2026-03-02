#! /bin/bash
exec &> stdout_plot

main() {
    
# Model name
name=t4

# Set plot size and properties
R='-R0/2000/-10/700'; J='-JX6.0i/-1.65i';  B='-BWS -Bx250f100 -By100f50';



# Define time steps 
steps=( 25 )

# Define parameters to plot
params=( viscosity )

# Loop through time steps
for step in ${steps[@]}; do

    # Loop through parameters
    for param in ${params[@]}; do 
        
        # Define pdf file name
        pdffile=$name'.'$step'.'$param

        # Define data file name
        infile='E:/backup/DoubleSubduction/model/double_plstc_subeen/gwb_add/ts/t4/output_t4/solution/out/'$name'.'$step'.'$param'.out'

        # Define plot position and plot basemap
        P='-X+0.5i -Y+8.40i'; 
        
        gmt begin $pdffile pdf

        # Set general gmt parameters
        # set_gmt_general_parameters

        gmt basemap $J $R $B $P

        color_scale # Make color scale
        gmt plot -A -L -Ccolor.cpt  $infile 
        color_scale # Make color scale for temperature contour plot

        # Plot color bar
        plot_colorbar

        # Plot figure label
        plot_figure_labels
        
        gmt end show
    done
done
}


function plot_figure_labels() {

# Read in model time
myr=$(head -n 1 $infile)
myr2=$(printf "%4.1f" $myr) # round value to have only 1 significant digit


echo   5  -25  8  0 ML  $myr2 'Myr'| gmt text  -F+f+a+j -N 
#echo   5  -25  8  0 ML  $myr2 'Myr'| gmt text  -F+f+a+j -N -Gblack   (make note black)
echo -40  1100  8  0 MC '(km)'      | gmt text  -F+f+a+j -N  
echo 250   -5  8   0 MC $label     | gmt text  -F+f+a+j -N 

}

function color_scale() {


if [ "$param" == "pres" ] ; then
  gmt makecpt -H -T0/4000/100 -Z  > color.cpt
   unit='Pressure (MPa)'; L=' -B500'
fi

if [ "$param" == "srrt" ] ; then
  nfb="--COLOR_NAN=- --COLOR_BACKGROUND=- --COLOR_FOREGROUND=-"
  gmt makecpt -H -T-20/-13/0.1 -Crainbow -Z > color.cpt
   unit='Logarithmic Viscosity (Pa s)'; L=' -B1'
fi

if [ "$param" == "temp" ] ; then
  gmt makecpt -H -T200/1600/100 -Crainbow -Z  > color.cpt
  S='--D_FORMAT=%4.f'; unit='Temperature (K)'; L=' -B200'
fi

if [ "$param" == "viscosity" ] ; then
  gmt makecpt -H -T20/26/0.1 -Cseis -Z -M $nfb > color.cpt
   unit='Logarithmic Viscosity (Pa s)'; L=' -B1'
fi

if [ "$param" == "OceanicMantle" ] ; then
  echo "1   173  150  122    2  173  150  122"  > color.cpt
  echo "2   197  140  121    3  197  140  121" >> color.cpt
  echo "3   119  196  132    4  119  196  132" >> color.cpt
  echo "4   255  255  255    5  255  255  255" >> color.cpt
   unit='Logarithmic Viscosity (Pa s)'; L=' -B1'
fi


}

function set_gmt_general_parameters() {


gmt set LC_NUMERIC "en_US.UTF-8" # fix for printing issue that came up with $sec variable

}

function plot_colorbar() {

gmt colorbar -DjMR+w2c/0.5c+o-1c/0c -Ccolor.cpt $L
echo 1000 900 9 0 CM $unit |gmt text -F+f+a+j -N

}



main "$@"

# rm -f color.cpt