function [] = T_mul(X,Y,data_dir,data_tempdir,l3)
% 20200903, liumengxue-THU, 
% 2.Calculate Temperature structure.
% continent-ocean-arc(microcontinent)-ocean-continent
% Top = 0 Celsius, 
% Moho= 500(Continent)
% Bottom of lithosphere = 1300 Celsius.
% Three kind of Temperature structures:
% 1)weak zone + continent plate.  -- Continental Plate
% 2)sediment + oceanic plate.     -- Oceanic Plate (cooling model)
% 3)mantle.                       -- mantle(0.5K/km, linear)
% Coordinate transformation:
% Depth read from file: bottom, y=0;
%                       top   , y=max(Y);
% Change depth to     : bottom, y=max(Y);
%                       top   , y=0;
% when change size of the model, 
% Remember to change dz_cl=[*,] and dz=[*,].(thickness of plates)
%
%
tic   % clock on
xsize   = max(X);     % length of model
ysize   = max(Y);     % depth of model 
%====================================================
%==============T of Left (Indian) Craton continental plate================
%----left continent------
fprintf('== Temperature of Continental plate ...\n');
filenamecp1_l=strcat(data_tempdir,'C1-left continental upper crust.txt');
cplate1_l=load(filenamecp1_l);   % 
filenamecp2_l=strcat(data_tempdir,'C2-left continental lower crust.txt');
cplate2_l=load(filenamecp2_l);   % 
filename3_l=strcat(data_tempdir,'C3-left continental lithospheric mantle.txt');
cplate3_l=load(filename3_l);     %      
cplate_l = [cplate1_l,cplate2_l,cplate3_l];  % 
% Input Parameters
k_cl  = [2.5,2.5];        % Thermal conductivity
dz_cl = [35.e3,85.e3];    % Layer thickness (m), curst, lithospheric mantle
A_cl  = [1.e-6,0.];       % Radiogenic heat production (W/m^3)
Tt_c  = [273.,823.];      % Temperature at top of layer(K)
Tb_cl = [823.,1573];      % Temperature at base of layer(K)
qt_cl = [0.,0.];          % Heat flow at top of layer
qb_cl = [0.,0.];          % Heat flow at base of layer
% 
qt_cl(1) = ( Tb_cl(1) - Tt_c(1) + (A_cl(1)*dz_cl(1)^2)/(2.*k_cl(1)) )*k_cl(1)/dz_cl(1);       
qb_cl(1) = qt_cl(1) - A_cl(1)*dz_cl(1);  % crustal basal heat flow
qt_cl(2) = qb_cl(1);                   % top lithospheric mantle heat flow 
% Determine lithospheric mantle thermal conductivity
k_cl(2)=(qt_cl(2)*dz_cl(2) - 0.5*A_cl(2)*dz_cl(2)^2)/(Tb_cl(2)-Tt_c(2));
% Determine lithosphere mantle basal heat flow # 
qb_cl(2) = qt_cl(2) - A_cl(2)*dz_cl(2);
% change depth to: bottom, y=max(Y);
%                  top, y=0;
z_cl = ysize*ones(1,length(cplate1_l)+length(cplate2_l)+length(cplate3_l))-cplate_l(2,:); %z:depth,change surface z=600km to z=0;
[m_cl,n_cl]=size(z_cl);
for i=1:n_cl
    if z_cl(i)<=dz_cl(1)
       t_cl(i) = Tt_c(1) + (qt_cl(1)/k_cl(1))*z_cl(i) - (A_cl(1)*(z_cl(i)^2))/(2*k_cl(1));
    else
       t_cl(i) = Tt_c(2) + (qt_cl(2)/k_cl(2))*(z_cl(i)-dz_cl(1)) - (A_cl(2)*((z_cl(i)-dz_cl(1))^2))/(2*k_cl(2)); 
    end
end
fprintf('== Done! == ...\n');
%
%----Middle(South China) continent------
fprintf('== Temperature of Continental plate ...\n');
filenamecp1_m=strcat(data_tempdir,'C8-middle upper crust.txt');
cplate1_m=load(filenamecp1_m);   % 
filenamecp2_m=strcat(data_tempdir,'C9-middle lower crust.txt');
cplate2_m=load(filenamecp2_m);   % 
filename3_m=strcat(data_tempdir,'C10-middle continental lithospheric mantle.txt');
cplate3_m=load(filename3_m);     % 
cplate_m = [cplate1_m,cplate2_m,cplate3_m];  % 
% Input Parameters
k_cm  = [2.5,2.5];        % Thermal conductivity
dz_cm = [35.e3,85.e3];   % Layer thickness (m), curst, lithospheric mantle
A_cm  = [1.e-6,0.];       % Radiogenic heat production (W/m^3)
Tt_c  = [273.,823.];      % Temperature at top of layer(K)
Tb_cm = [823.,1573];      % Temperature at base of layer(K)
qt_cm = [0.,0.];          % Heat flow at top of layer
qb_cm = [0.,0.];          % Heat flow at base of layer
% 
qt_cm(1) = ( Tb_cm(1) - Tt_c(1) + (A_cm(1)*dz_cm(1)^2)/(2.*k_cm(1)) )*k_cm(1)/dz_cm(1);       
qb_cm(1) = qt_cm(1) - A_cm(1)*dz_cm(1);  % crustal basal heat flow
qt_cm(2) = qb_cm(1);                   % top lithospheric mantle heat flow 
% Determine lithospheric mantle thermal conductivity
k_cm(2)=(qt_cm(2)*dz_cm(2) - 0.5*A_cm(2)*dz_cm(2)^2)/(Tb_cm(2)-Tt_c(2));
% Determine lithosphere mantle basal heat flow # 
qb_cm(2) = qt_cm(2) - A_cm(2)*dz_cm(2);
% change depth to: bottom, y=max(Y);
%                  top, y=0;
z_cm = ysize*ones(1,length(cplate1_m)+length(cplate2_m)+length(cplate3_m))-cplate_m(2,:); %z:depth,change surface z=600km to z=0;
[m_cm,n_cm]=size(z_cm);
for i=1:n_cm
    if z_cm(i)<=dz_cm(1)
       t_cm(i) = Tt_c(1) + (qt_cm(1)/k_cm(1))*z_cm(i) - (A_cm(1)*(z_cm(i)^2))/(2*k_cm(1));
    else
       t_cm(i) = Tt_c(2) + (qt_cm(2)/k_cm(2))*(z_cm(i)-dz_cm(1)) - (A_cm(2)*((z_cm(i)-dz_cm(1))^2))/(2*k_cm(2)); 
    end
end
fprintf('== Done! == ...\n');
%
%---- Right continental plate (South Sea) ------
%% ====== if SS is ocean, comment these section=======
% filenamecp1_r=strcat(data_tempdir,'C11-right continental upper crust.txt');
% cplate1_r=load(filenamecp1_r);   % 
% filenamecp2_r=strcat(data_tempdir,'C12-right continental lower crust.txt');
% cplate2_r=load(filenamecp2_r);   % 
% filename3_r=strcat(data_tempdir,'C13-right continental lithospheric mantle.txt');
% cplate3_r=load(filename3_r);     % 
% cplate_r = [cplate1_r,cplate2_r,cplate3_r];  % 
% % Input Parameters
% k_cr  = [2.5,2.5];        % Thermal conductivity
% dz_cr = [35.e3,85.e3];    % Layer thickness (m), curst, lithospheric mantle
% A_cr  = [1.e-6,0.];       % Radiogenic heat production (W/m^3)
% Tt_c  = [273.,823.];      % Temperature at top of layer(K)
% Tb_cr = [823.,1573];      % Temperature at base of layer(K)
% qt_cr = [0.,0.];          % Heat flow at top of layer
% qb_cr = [0.,0.];          % Heat flow at base of layer
% % 
% qt_cr(1) = ( Tb_cr(1) - Tt_c(1) + (A_cr(1)*dz_cr(1)^2)/(2.*k_cr(1)) )*k_cr(1)/dz_cr(1);       
% qb_cr(1) = qt_cr(1) - A_cr(1)*dz_cr(1);  % crustal basal heat flow
% qt_cr(2) = qb_cr(1);                   % top lithospheric mantle heat flow 
% % Determine lithospheric mantle thermal conductivity
% k_cr(2)=(qt_cr(2)*dz_cr(2) - 0.5*A_cr(2)*dz_cr(2)^2)/(Tb_cr(2)-Tt_c(2));
% % Determine lithosphere mantle basal heat flow # 
% qb_cr(2) = qt_cr(2) - A_cr(2)*dz_cr(2);
% % change depth to: bottom, y=max(Y);
% %                  top, y=0;
% z_cr = ysize*ones(1,length(cplate1_r)+length(cplate2_r)+length(cplate3_r))-cplate_r(2,:); %z:depth,change surface z=600km to z=0;
% [m_cr,n_cr]=size(z_cr);
% for i=1:n_cr
%     if z_cr(i)<=dz_cr(1)
%        t_cr(i) = Tt_c(1) + (qt_cr(1)/k_cr(1))*z_cr(i) - (A_cr(1)*(z_cr(i)^2))/(2*k_cr(1));
%     else
%        t_cr(i) = Tt_c(2) + (qt_cr(2)/k_cr(2))*(z_cr(i)-dz_cr(1)) - (A_cr(2)*((z_cr(i)-dz_cr(1))^2))/(2*k_cr(2)); 
%     end
% end
% cplateT=[cplate_l,cplate_m,cplate_r;t_cl,t_cm,t_cr];
%% ==========================================================
cplateT=[cplate_l,cplate_m;t_cl,t_cm];
filenameopT=strcat(data_tempdir,'T of continental plate.txt');
dlmwrite(filenameopT,cplateT,'delimiter','\t','precision','%8.6f');
%
[mCP,nCP]=size(cplateT);

%==================== Oceanic Plate =================================
%======== T of left oceanic plate (Neo-Tethys) ================
fprintf('==Temperature of Neo-Tethys ...\n');
filenameop1=strcat(data_tempdir,'C4-left sediment.txt');
ocean1=load(filenameop1);  %
filenameop2=strcat(data_tempdir,'C5-left oceanic crust.txt');
ocean2=load(filenameop2);  % 
filenameop3=strcat(data_tempdir,'C6-left oceanic lithospheric mantle.txt');
ocean3=load(filenameop3);  % 
filenameop4=strcat(data_tempdir,'C7-left weak zone.txt');
ocean4=load(filenameop4);  %
leftocean = [ocean1,ocean2,ocean3,ocean4]; %
% Input Parameters
knt  = [2.5,83.2]; % Thermal conductivity
dznt = [90.e3,0];  % oceanic thickness(m)(sediment+crust+lithospheric mantle)
Ttnt = [273.,1573];% Temperature at top of layer(K)
Tbnt = [1573.,0];  % Temperature at base of layer(K)
agent= 70;                 % Age of the oceanic plate(Ma:Million year)
rhont= [2900,3370];        % Density (kg/m^3)
cpnt = [750,750];          % Specific heat(J/kg*K)
% 
age_snt=agent*1.e6*365*24*60*60; % change unit 'Ma' into 's(second)';
kaint= knt./(rhont.*cpnt); 
%  
% z:depth,change surface z=600km to z=0; 
% change depth to: bottom, y=max(Y);
%                  top, y=0;
znt = ysize*ones(1,length(ocean1)+length(ocean2)+length(ocean3)+...
    length(ocean4))-leftocean(2,:);
zmaxnt=max(znt);
zmin=min(znt);
[mnt,nnt]=size(znt);      %
bdepnt =znt(nnt);         % depth of model
for i=1:nnt
       %Tdep(i) = Tt(1)+(Tb(1)-Tt(1))*erf(z(i)/(2*sqrt(kai(1)*age_s)));     
       Tdepleft(i) = Ttnt(1)+(Tbnt(1)-Ttnt(1))*znt(i)/dznt(1);     

end
fprintf('== Done! == ...\n');
%% ==============================================================
%======== T of South Sea oceanic plate ================
% If SS is cotinent, comment this section 
fprintf('==Temperature of South Sea Oceanic Plate ...\n');
fprintf('== Done! == ...\n');
filenameop5=strcat(data_tempdir,'C11-south sea sedment.txt');
ocean5=load(filenameop5);  %
filenameop6=strcat(data_tempdir,'C12-south sea crust.txt');
ocean6=load(filenameop6);  % 
filenameop7=strcat(data_tempdir,'C13-south sea lithospheric mantle.txt');
ocean7=load(filenameop7);  % 
filenameop8=strcat(data_tempdir,'C14-vertical weak zone.txt');
ocean8=load(filenameop8);  %
ssocean = [ocean5,ocean6,ocean7,ocean8]; %
% Input Parameters
kss  = [2.5,83.2]; % Thermal conductivity
dzss = [60.e3,0];  % oceanic thickness(m)(sediment+crust+lithospheric mantle)
Ttss = [273.,1573];% Temperature at top of layer(K)
Tbss = [1573.,0];  % Temperature at base of layer(K)
agess= 70;                 % Age of the oceanic plate(Ma:Million year)
rhoss= [2900,3370];        % Density (kg/m^3)
cpss = [750,750];          % Specific heat(J/kg*K)
% 
age_s_ss=agess*1.e6*365*24*60*60; % change unit 'Ma' into 's(second)';
kaiss= kss./(rhoss.*cpss); 
%  
% z:depth,change surface z=600km to z=0; 
% change depth to: bottom, y=max(Y);
%                  top, y=0;
zss = ysize*ones(1,length(ocean5)+length(ocean6)+length(ocean7)+...
    length(ocean8))-ssocean(2,:);
zmaxss=max(zss);
zminss=min(zss);
[mss,nss]=size(zss);      %
bdepss =zss(nss);         % depth of model
for i=1:nss
       %Tdep(i) = Tt(1)+(Tb(1)-Tt(1))*erf(z(i)/(2*sqrt(kai(1)*age_s)));     
       Tdepss(i) = Ttss(1)+(Tbss(1)-Ttss(1))*zss(i)/dzss(1);     

end
%%
%======== T of right oceanic plate (Philippine Sea)================ 
fprintf('==Temperature of Philippine Sea Oceanic Plate ...\n')
filenameop9=strcat(data_tempdir,'C15-right sediment.txt');
ocean9=load(filenameop9);  %
filenameop10=strcat(data_tempdir,'C16-right crust.txt');
ocean10=load(filenameop10);  % 
filenameop11=strcat(data_tempdir,'C17-right lithospheric mantle.txt');
ocean11=load(filenameop11);  % 
filenameop12=strcat(data_tempdir,'C18-right weak zone.txt');
ocean12=load(filenameop12);  %
psocean = [ocean9,ocean10,ocean11,ocean12]; %
% Input Parameters
kps  = [2.5,83.2]; % Thermal conductivity
dzps = [60.e3,0];  % oceanic thickness(m)(sediment+crust+lithospheric mantle)
Ttps = [273.,1573];% Temperature at top of layer(K)
Tbps = [1573.,0];  % Temperature at base of layer(K)
ageps= 70;                 % Age of the oceanic plate(Ma:Million year)
rhops= [2900,3370];        % Density (kg/m^3)
cpps = [750,750];          % Specific heat(J/kg*K)
% 
age_s_ps=ageps*1.e6*365*24*60*60; % change unit 'Ma' into 's(second)';
kaips= kps./(rhops.*cpps); 
%  
% z:depth,change surface z=600km to z=0; 
% change depth to: bottom, y=max(Y);
%                  top, y=0;
zps = ysize*ones(1,length(ocean9)+length(ocean10)+length(ocean11)+...
    length(ocean12))-...
    psocean(2,:);
zmaxps=max(zps);
zminps=min(zps);
[mps,nps]=size(zps);      %
bdepps =zps(nps);         % depth of model
for i=1:nps
       %Tdep(i) = Tt(1)+(Tb(1)-Tt(1))*erf(z(i)/(2*sqrt(kai(1)*age_s)));     
       Tdepps(i) = Ttps(1)+(Tbps(1)-Ttps(1))*zps(i)/dzps(1);     

end
%-------
%======== T of right right oceanic plate (Pacific Ocean)================ 
fprintf('==Temperature of Pacific Ocean Plate ...\n')
filenameop13=strcat(data_tempdir,'C19-right right sediment.txt');
ocean13=load(filenameop13);  %
filenameop14=strcat(data_tempdir,'C20-right right crust.txt');
ocean14=load(filenameop14);  % 
filenameop15=strcat(data_tempdir,'C21-right right lithospheric mantle.txt');
ocean15=load(filenameop15);  % 
paocean = [ocean13,ocean14,ocean15]; %
% Input Parameters
kpa  = [2.5,83.2]; % Thermal conductivity
dzpa = [90.e3,0];  % oceanic thickness(m)(sediment+crust+lithospheric mantle)
Ttpa = [273.,1573];% Temperature at top of layer(K)
Tbpa = [1573.,0];  % Temperature at base of layer(K)
agepa= 70;                 % Age of the oceanic plate(Ma:Million year)
rhopa= [2900,3370];        % Density (kg/m^3)
cppa = [750,750];          % Specific heat(J/kg*K)
% 
age_s_pa=agepa*1.e6*365*24*60*60; % change unit 'Ma' into 's(second)';
kaipa= kpa./(rhopa.*cppa); 
%  
% z:depth,change surface z=600km to z=0; 
% change depth to: bottom, y=max(Y);
%                  top, y=0;
zpa = ysize*ones(1,length(ocean13)+length(ocean14)+length(ocean15))-paocean(2,:);
zmaxpa=max(zpa);
zminpa=min(zpa);
[mpa,npa]=size(zpa);      %
bdeppa =zpa(npa);         % depth of model
for i=1:npa
       %Tdep(i) = Tt(1)+(Tb(1)-Tt(1))*erf(z(i)/(2*sqrt(kai(1)*age_s)));     
       Tdeppa(i) = Ttpa(1)+(Tbpa(1)-Ttpa(1))*zpa(i)/dzpa(1);     

end
%------------
% Tdepmax =max(Tdep);
% Tdepmin =min(Tdep);
oplateT=[leftocean,ssocean,psocean,paocean;Tdepleft,Tdepss,Tdepps,Tdeppa];
filenameopT=strcat(data_tempdir,'T of oceanic plate.txt');
dlmwrite(filenameopT,oplateT,'delimiter','\t','precision','%8.6f');
%
%===================================================
%========Temperature of mantle ==========
% Temperature gradient in the mantle below continent is 5K/km;
Ttm=1573;  % temperature on bottom of lithosphere(1300Celsius) 
fprintf('==Temperature of Mantle ...\n');
filenameman=strcat(data_tempdir,'C22-mantle.txt');
mantle1=load(filenameman); % 
man2=mantle1(2,:);
% change depth to: bottom, y=max(Y);
%                  top, y=0;
mantledep= ysize*ones(1,length(man2))-man2;
dept_man=dz_cl(1)+dz_cl(2);  %top depth of mantle(thickness of cotinental plate)
%
[mman,nman]=size(mantledep);
for i=1:nman
     mant(i)= Ttm + 0.5*((mantledep(i)-dept_man)/1.e3);  % T in mantle 0.5K/km("Geodynamics P223")
end
max_mantle = max(mant);
mantle=[mantle1;mant]; % mantle coordinate with temperature, 
filename=strcat(data_tempdir,'T of mantle.txt'); 
dlmwrite(filename,mantle,'delimiter','\t','precision','%8.6f');
total=[cplateT,oplateT,mantle]';  % 3 colomn
xsort=sortrows(total,1);
ysort=sortrows(xsort,2);
Tsortxy=ysort;
fprintf('--Saving temperature data ...\n');
filenametotal=strcat(data_dir,'temperature.txt'); 
fid=fopen(filenametotal,'w');
l1='# Test data for ascii data initial conditions.';
l2='# Only next line is parsed in format: [nx] [ny] because of keyword "POINTS:"';
%l3='# POINTS: 1001 331';
l4='# Columns: x y temperature [K]';
fprintf(fid,'%s\n',l1,l2,l3,l4);
fclose(fid);
dlmwrite(filenametotal,Tsortxy,'-append','delimiter','\t','precision','%8.6f');
%dlmwrite(filenametotal,Tsortxy,'-append','delimiter','\t','precision','%6.0f');
%
% Plot can't use file with header.
Ttemp=strcat(data_tempdir,'temperature.txt');
dlmwrite(Ttemp,Tsortxy,'delimiter','\t','precision','%8.6f');
%
fprintf('Done!\n');
toc   %clock off
end