# ---2025-3-26-lhl-THU--
#
# Load modules
import numpy as np
import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy

# Time steps to analyze
step_start = 25
step_end   = 25
step_int   = 1   # time interval

# Model name
name = 't4'

# Path to 'solution' directory with .pvtu files
# folder = 'E:/backup/DoubleSubduction/model/double_plstc_subeen/gwb/PA200km/t100/v8t100/solution/'
folder = 'E:/backup/DoubleSubduction/model/double_plstc_subeen/gwb_add/ts/t4/output_t4/solution/'

save_folder = 'E:/backup/DoubleSubduction/model/double_plstc_subeen/gwb_add/ts/t4/output_t4/solution/out/'

#
output_interval = 1.e5  # units: year/yr

# Types of data to extract
fields = ["viscosity","T","density","strain_rate"]
# fields = ["density"]
# fields = ["viscosity","OceanicMantle","density","Sediment"]

# y-coordinate of vertical slices to extract (y=0 is front face)
#y_slices = [0.e3]

# Loop through time steps
for step in range(step_start,step_end+1,step_int):
    
    # String for output file number
    if step<10.:
        step_string = '0000' + str(step)
    elif step>=10 and step<100:
        step_string = '000' + str(step)
    else:
        step_string = '00' + str(step)
        
    # Set model file name, including path to 'solution' directory
    file_name = folder + 'solution-' + step_string + '.pvtu'
    print(file_name)

    # Load vtu data (pvtu directs to vtu files)
    reader = vtk.vtkXMLPUnstructuredGridReader()
    print(reader)
    reader.SetFileName(file_name)
    reader.Update()

    # Get the coordinates of nodes in the mesh
    nodes_vtk_array= reader.GetOutput().GetPoints().GetData()
                
    # Convert nodal vtk data to a numpy array
    nodes_numpy_array = vtk_to_numpy(nodes_vtk_array)
                
    # Extract x, y and z coordinates from numpy array
    x,y = nodes_numpy_array[:,0] , nodes_numpy_array[:,1]
                    
    # Loop through fields
    for field in fields:
        
        # Extract field value
        field_vtk_array = reader.GetOutput().GetPointData().GetArray(field)
        field_numpy_array = vtk_to_numpy(field_vtk_array)
            
        # Modify field values
       
        if field =="viscosity":
            v = np.log10(field_numpy_array)
        # ---20190813-lmx-THU---
        else:
            v = field_numpy_array
        # ------

        # Indices on vertical slice parallel to y-direction
        #for y_slice in y_slices:
                                
            # Indicies for current y_slice
            #s = np.where(y[:]==y_slice)

        # Open data out file
        if os.path.exists(save_folder):
            out_file = open(save_folder + name + '.' + str(step) + '.' + field + '.out', 'w')
        else:
            os.makedirs(save_folder)
            out_file = open(save_folder + name + '.' + str(step) + '.' + field + '.out', 'w')
        # Write time to outfile in Myr(Million years)
        out_file.write('%-10.4f\n' % (step*output_interval/1.e6))
                                
        # Loop through points and write out data for cells according to GMT formatting
        # Cordinates and field values of 4 cells are represented with 9 points.
        # Node order for each cell: 00, 10, 01, 11
        # ------20190814-lmx-THU-----
        # Figure of 4 cells represented with 9 points:
        #     6      7      8
        #     _______________
        #     |cell3 |cell4 |
        #   3 |______|4_____| 5
        #     |cell1 |cell2 |
        #     |______|______|
        #     0      1      2
        # --------------------------
        for i in range(0,v.size,9):
            # Cell 1 (bottom left)
            x_c1 = np.array([x[i+0],x[i+1],x[i+3],x[i+4]])
            y_c1 = np.array([y[i+0],y[i+1],y[i+3],y[i+4]])
            v_c1 = np.array([v[i+0],v[i+1],v[i+3],v[i+4]])
            # ----20190814-lmx-THU
            # counterclockwise point0,1,4,3-> xc1[0],xc1[1],xc1[3],xc1[2]
            out_file.write('> -Z %-8.4f\n' % (np.mean(v_c1)))
            out_file.write('%8.4f %8.4f\n' % (x_c1[0]/1.e3,(y_c1[0] - 700.e3)/-1.e3))
            out_file.write('%8.4f %8.4f\n' % (x_c1[1]/1.e3,(y_c1[1] - 700.e3)/-1.e3))
            out_file.write('%8.4f %8.4f\n' % (x_c1[3]/1.e3,(y_c1[3] - 700.e3)/-1.e3))
            out_file.write('%8.4f %8.4f\n' % (x_c1[2]/1.e3,(y_c1[2] - 700.e3)/-1.e3))
            out_file.write('%8.4f %8.4f\n' % (x_c1[0]/1.e3,(y_c1[0] - 700.e3)/-1.e3))
                    
            # Cell 2 (bottom rigt)
            x_c2 = np.array([x[i+1],x[i+2],x[i+4],x[i+5]])
            y_c2 = np.array([y[i+1],y[i+2],y[i+4],y[i+5]])
            v_c2 = np.array([v[i+1],v[i+2],v[i+4],v[i+5]])
            #
            out_file.write('> -Z %-8.4f\n' % (np.mean(v_c2)))
            out_file.write('%8.4f %8.4f\n' % (x_c2[0]/1.e3,(y_c2[0] - 700.e3)/-1.e3))
            out_file.write('%8.4f %8.4f\n' % (x_c2[1]/1.e3,(y_c2[1] - 700.e3)/-1.e3))
            out_file.write('%8.4f %8.4f\n' % (x_c2[3]/1.e3,(y_c2[3] - 700.e3)/-1.e3))
            out_file.write('%8.4f %8.4f\n' % (x_c2[2]/1.e3,(y_c2[2] - 700.e3)/-1.e3))
            out_file.write('%8.4f %8.4f\n' % (x_c2[0]/1.e3,(y_c2[0] - 700.e3)/-1.e3))
                
            # Cell 3 (top left)
            x_c3 = np.array([x[i+3],x[i+4],x[i+6],x[i+7]])
            y_c3 = np.array([y[i+3],y[i+4],y[i+6],y[i+7]])
            v_c3 = np.array([v[i+3],v[i+4],v[i+6],v[i+7]])
            #
            out_file.write('> -Z %-8.4f\n' % (np.mean(v_c3)))
            out_file.write('%8.4f %8.4f\n' % (x_c3[0]/1.e3,(y_c3[0] - 700.e3)/-1.e3))
            out_file.write('%8.4f %8.4f\n' % (x_c3[1]/1.e3,(y_c3[1] - 700.e3)/-1.e3))
            out_file.write('%8.4f %8.4f\n' % (x_c3[3]/1.e3,(y_c3[3] - 700.e3)/-1.e3))
            out_file.write('%8.4f %8.4f\n' % (x_c3[2]/1.e3,(y_c3[2] - 700.e3)/-1.e3))
            out_file.write('%8.4f %8.4f\n' % (x_c3[0]/1.e3,(y_c3[0] - 700.e3)/-1.e3))
            
            # Cell 4 (top right)
            x_c4 = np.array([x[i+4],x[i+5],x[i+7],x[i+8]])
            y_c4 = np.array([y[i+4],y[i+5],y[i+7],y[i+8]])
            v_c4 = np.array([v[i+4],v[i+5],v[i+7],v[i+8]])
            #
            out_file.write('> -Z %-8.4f\n' % (np.mean(v_c4)))
            out_file.write('%8.4f %8.4f\n' % (x_c4[0]/1.e3,(y_c4[0] - 700.e3)/-1.e3))
            out_file.write('%8.4f %8.4f\n' % (x_c4[1]/1.e3,(y_c4[1] - 700.e3)/-1.e3))
            out_file.write('%8.4f %8.4f\n' % (x_c4[3]/1.e3,(y_c4[3] - 700.e3)/-1.e3))
            out_file.write('%8.4f %8.4f\n' % (x_c4[2]/1.e3,(y_c4[2] - 700.e3)/-1.e3))
            out_file.write('%8.4f %8.4f\n' % (x_c4[0]/1.e3,(y_c4[0] - 700.e3)/-1.e3))
                
# Close data out file
out_file.close()


