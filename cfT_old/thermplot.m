function [] = thermplot(data_tempdir)
% 2. Same as .../c7-vertical/isotherm.m
% Plot isotherm via ascii data 
fprintf('====Begin to plot isotherm======\n')
tic
filename=strcat(data_tempdir,'Temperature.txt');
total=load(filename);
fprintf('==Running...');
x=total(:,1);
y=total(:,2);
z=total(:,3);
[X,Y]=meshgrid(min(x):2.e3:max(x),min(y):2.e3:max(y));
Z= griddata(x,y,z,X,Y);
figure('NumberTitle', 'off', 'Name', 'Isotherm');
%
[c,h]=contour(X,Y,Z);
clabel(c,h);
%gtext('Isotherm');
xlabel('Length(km)');
ylabel('Depth(km)');

toc   %clock off
%
fprintf('Finished\n');
%
end
