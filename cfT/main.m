% 20250414,ies,mt/mt*.prm
% Main mfile to create compositional fields, calculate initial 
% temperature field, and then save the data above for ASPECT input.
% 1.Change name to your own folder to save data
% 2.Run main.m
% 3.When change X, change len_cp and len_op in cfwh.h and cfsp.m
%   When change Y, change dz_c=[*,] and dz=[*,] in Ts.m(layers thickness)
% Schematic figure of the model
% More details, see each function.
% Model size 4000km*660km, user-defined; 
% See each file for other user-defined input parameters.
% Copyright: MengxueLiu-ies, 20250414
% include strain weakening
% t7:SS=60km; 
% t8:SS=80km;
% t11:SS=50km;
% ******
%
clear all
clc
mkdir('E:\backup\DoubleSubduction\model\double_plstc_subeen\gwb_add\ts\t11\temp\');
data_dir = 'E:\backup\DoubleSubduction\model\double_plstc_subeen\gwb_add\ts\t11\'; % folder for compositional field
data_tempdir = 'E:\backup\DoubleSubduction\model\double_plstc_subeen\gwb_add\ts\t11\temp\';% template files
X=0:2.e3:4000.e3;         % length of the model
Y=0:2.e3:660.e3;          % width of the model
l3='# POINTS: 2001 331';  % nx,ny
cf_mul(X,Y,data_dir,data_tempdir,l3); 
T_mul(X,Y,data_dir,data_tempdir,l3);
thermplot(data_tempdir);
rmdir('E:\backup\DoubleSubduction\model\double_plstc_subeen\gwb_add\ts\t11\temp','s');% delete folder and files
