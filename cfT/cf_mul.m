function [] = cf_mul(X,Y,data_dir,data_tempdir,l3)
% 2020,liumengxue-THU
% 1. Save compositional fields data for ASPECT
% continent-ocean-arc(microcontinent)-ocean-continent
% free surface.
% Two type: with OR without weak zone between ocean-arc.
% Model Schematic Diagram
% IndianPlate - NeoTethys - South China - South Sea - Philippine Sea - Pacific Ocean 
% __________________________________________________________________________________________
% |            |        \   \            |            |  |              /  /                |
% |Continent   |Ocean1   \   \ Continent2| ocean2/Con |  |    Ocean3   /  /     PA Ocean    |
% |  120(km)   |100(km)   \ wz\ 120(km)  | 40(km) - ? |wz|     60km   /wz/       90(km)     |
% | 20+15+85   |4+8+88     \   \20+15+85 | 4+8+varies |  |    4+8+48 /  /        4+8+78     |      
% |            |____________\___\        |____________|__|__________/__/                    |
% |____________|                 \_______|                            /_____________________|
% |                                                                                         |
% |                                                                                         |
% |                                                                                         |
% |                                                                                         |
% |                                                                                         |
% |_________________________________________________________________________________________|
%
% Model size 4000km * 660km;
% Continental plate 120km,
% Oceanic plate 40 - 90km;
%----------------Unit:km---------------------------------------------------
%******* Continent 1 ***********
%       left Continenal upper crust             |  20  |C1
%       left Continenal lower crust             |  15  |C2
%       left Continenal lithospheric mantle     |  85  |C3
%       ledt Continenal plate                   |  120 |
%******* Ocean 1 *********
%       left Sediment                           |  4   |C4
%       left Oceanic crust                      |  8   |C5
%       left Oceanic lithospheric mantle        |  78  |C6
%       left Oceanic plate                      |  90  |
%******* Weak Zone 1 *********
%       Right inclined                          |  90  |C7
%******* Continent 2(South China) ****
%       Continenal upper crust                  |  20  |C8
%       Continenal lower crust                  |  15  |C9
%       Continenal lithospheric mantle          |  85  |C10
%       Continenal plate                        |  120 |
%******* Ocean 2 / Continent2 *********
%       right Sediment                          |  4   |C11
%       right Oceanic crust                     |  8   |C12
%       right Oceanic lithospheric mantle       |  88  |C13
%       right Oceanic plate                     | 40-90|Varies
%******* Weak Zone 2 *********
%        Vertical                               | 40-90|C14
%******* Ocean 3 *****
%       right Sediment                          |  4   |C15
%       right Oceanic crust                     |  8   |C16
%       right Oceanic lithospheric mantle       |  48  |C17
%       right Oeanic plate                      |  60  |
%******* Weak Zone 3 *********
%       Right inclined                          |  90  |C18
%******* Ocean 4 *****
%       right right sediment                    |  4   |C19
%       right right oceanic crust               |  8   |C20
%       right right oceanic lithospheric mantle |  48  |C21
%       right Continenal plate                  |  90  |
%******* Plastic Strain *****
%         PS                                    | 120km|C22
%***********************************************
%       Mantle                                  |background|C23
%************************************************************
%       Length of Continenal plate              |  800 |
%       Length of Oceanic plate                 |  390 |
%       Length of Arc(microcontinent)           |  100 | (variable)
%       Width of Weak Zone                      |  10  |
%--------------------------------------------------------------------------
% 23 compositional fields(include mantle)
% more compositional fields 1) change CFnum = +1;
%                           2) add codes as discribed in #1 and #2 below 
% change size of model, if
% 1)change Y, nothing else you need to change.
% 2)change X, you need to change len_cp and len_op.
tic   % clock on
fprintf('=====Begin======\n');
% ---------Input Parameters that can be changed with different conditions----
% ---- Continental Plate thickness ----
th_CUCl   = 25.e3;     % thickness of  left Continental Upper Crust
th_CLCl   = 10.e3;     % thickness of  left Continental Lower Crust
th_CLMl   = 85.e3;     % thickness of  left Continental Lithospheric Mantle
th_CUCm   = 25.e3;     % thickness of  middle Continental Upper Crust
th_CLCm   = 10.e3;     % thickness of  middle Continental Lower Crust
th_CLMm   = 85.e3;     % thickness of  middle Continental Lithospheric Mantle
%**** ---- if South Sea use Ocean Plate, comment the following 3 lines ---
% th_CUCSS   = 25.e3;     % thickness of  right Continental Upper Crust
% th_CLCSS   = 10.e3;     % thickness of  right Continental Lower Crust
% th_CLMSS   = 85.e3;     % thickness of  right Continental Lithospheric Mantle
%**** -------------------------
% -------- Oceanic Plate thickness ----------
th_sedl    = 4.e3;     % thickness of sendiment layer on left Oceanic Plate
th_OCl     = 8.e3;     % thickness of left Oceanic Crust
th_OLMl    = 78.e3;    % thickness of left Oceanic Lithospheric Mantle
th_WZl     = th_sedl+th_OCl+th_OLMl; % thinkness of slope Weak Zone
%**** ---- if South Sea use Continetal Plate, change the chickness and tempsrature sturcture ---
th_sedSS    = 4.e3;     % thickness of sendiment layer on right Oceanic Plate
th_OCSS     = 8.e3;     % thickness of right Oceanic Crust
th_OLMSS    = 38.e3;    % thickness of right Oceanic Lithospheric Mantle change this 38,48,58,68 for 50,60,70,80 SCCS thickness
%-----
th_WZSS     = th_sedSS+th_OCSS+th_OLMSS; % thinkness of slope Weak Zone
%**** --------------------------------------
th_sedr    = 4.e3;     % thickness of sediment layer on right Oceanic Plate
th_OCr     = 8.e3;     % thickness of right Oceanic Crust
th_OLMr    = 48.e3;    % thickness of right Oceanic Lithospheric Mantle
th_WZr     = th_sedr+th_OCr+th_OLMr; % thinkness of slope Weak Zone
th_sedrr    = 4.e3;     % thickness of sendiment layer on right right Oceanic Plate
th_OCrr     = 8.e3;     % thickness of right right Oceanic Crust
th_OLMrr    = 78.e3;    % thickness of right right Oceanic Lithospheric Mantle
% --------  Plate length ----------
len_CPl       = 200.e3;    % length of left continental plate
len_OPl       = 400.e3;    % length of left oceanic plate
len_WZl       = 10.e3;     % length of left slope Weak Zone(left)
len_CPm       = 1390.e3;   % length of right continental plate
len_OSS_CSS   = 500.e3;    % length of South Sea
len_WZm       = 10.e3;     % length of middle vertical Weak Zone(middle)
len_OPr       = 880.e3;    % length of right oceanic plate
len_WZr       = 10.e3;     % length of right slope Weak Zone(right)
len_OPrr      = 600.e3;    % length of right right oceanic plate
anglel        = 45;        % left subducting angle = 90 -angle;(unit:degree)
angler        = 45;        % right subducting angle = 90 -angle;(unit:degree)
%======================
CFnum         = 22;        % number of Compositional Field(include mantle)
%======================
CFnum2    = 1;         % cf for strain weakening
%======================
% left slope weak zone
% y=hx+b,that is x=(y-b)/k---linear equation,tand(angle),tan(radian/arc)
x2 = (len_CPl+len_OPl)+(th_sedl+th_OCl+th_OLMl)/tand(anglel);   
k1 = (max(Y)-(max(Y)-(th_sedl+th_OCl+th_OLMl)))/...
     (len_CPl+len_OPl-x2);         % slope(left weak zone) 
b1 = max(Y)-k1*(len_CPl+len_OPl);  % intercept
% 
% right slope weak zone
% y=hx+b,that is x=(y-b)/k---linear equation,tand(angle),tan(radian/arc)
x4 = (max(X)-len_OPrr)-(th_sedrr+th_OCrr+th_OLMrr)/tand(angler);   
k2 = (max(Y)-(max(Y)-(th_sedrr+th_OCrr+th_OLMrr)))/...
     (max(X)-len_OPrr-x4);         % slope(left weak zone) 
b2 = max(Y)-k2*(max(X)-len_OPrr);  % intercept
%
C  = zeros(length(Y),length(X));   % Color for plot; 
ct = ones(CFnum+1,1);  % Points's number in each fields=ct(i)-1(i=1,2,...,6);
%=============================
% cf for strain weakening
C2 = zeros(length(Y),length(X));   % Color for plot2;
ct2= ones(CFnum2+1,1);
%===========================================
% ---------add #1------------
% If CFnum > 5,add code:
% elseif (expression)
%     C(i,j)= cnumber;  % color number
%    % "*" stands for index of compositional fields
%     c*_length(ct(*))=X(j);  % x
%     c*_width(ct(*))=Y(i);   % y 
%     c*_coord=[c*_length;c*_width]; %
%     ct(*)=ct(*)+1;
%------------------------
h = waitbar(0,'Processing compositional fields...');  % wait bar
fprintf('--Processing\n');
for i=1:length(Y)
    waitbar(i/length(Y),h)
    for j=1:length(X)
        if (Y(i)>(max(Y)-th_CUCl) & ...
            Y(i)<=max(Y) & ...
            X(j)<len_CPl)
            C(i,j)=0.9;             % C1:left-continental upper crust
            c1_length(ct(1))=X(j);  % coordinate x
            c1_width(ct(1))=Y(i);   % coordinate y
            c1_coord=[c1_length;c1_width]; % point(x,y)
            ct(1)=ct(1)+1;
        elseif (Y(i)>(max(Y)-th_CUCl-th_CLCl) &...
                Y(i)<=(max(Y)-th_CUCl) & ...
                X(j)<len_CPl)
            C(i,j)=0.8;               % C2:left-continental lower crust 
            c2_length(ct(2))=X(j);    % x
            c2_width(ct(2))=Y(i);     % y 
            c2_coord=[c2_length;c2_width]; %
            ct(2)=ct(2)+1;
        elseif (Y(i)>=(max(Y)-th_CUCl-th_CLCl-th_CLMl) & ...
                Y(i)<=(max(Y)-th_CUCl-th_CLCl) & ...
                X(j)<len_CPl)
            C(i,j)=0.7;             % C3:left-continental lithospheric mantle
            c3_length(ct(3))=X(j);  % x
            c3_width(ct(3))=Y(i);   % y
            c3_coord=[c3_length;c3_width]; %
            ct(3)=ct(3)+1;
        elseif (Y(i)>(max(Y)-th_sedl) & ... % 
                Y(i)<=max(Y) & ...
                X(j)<(Y(i)-b1)/k1 &...
                X(j)>=len_CPl)
            C(i,j)=0.6;             % C4: left sediment
            c4_length(ct(4))=X(j);  % x
            c4_width(ct(4))=Y(i);   % y
            c4_coord=[c4_length;c4_width]; %
            ct(4)=ct(4)+1;
        elseif (Y(i)>(max(Y)-th_sedl-th_OCl) & ...
                Y(i)<=(max(Y)-th_sedl) &...
                X(j)<(Y(i)-b1)/k1 &...
                X(j)>=len_CPl)
            C(i,j)=0.5;             % C5:left oceanic crust
            c5_length(ct(5))=X(j);  % x
            c5_width(ct(5))=Y(i);   % y
            c5_coord=[c5_length;c5_width]; %
            ct(5)=ct(5)+1;     
        elseif (Y(i)>=(max(Y)-th_sedl-th_OCl-th_OLMl) & ...
                Y(i)<=(max(Y)-th_sedl-th_OCl) &...
                X(j)<(Y(i)-b1)/k1 &...
                X(j)>=len_CPl)
            C(i,j)=0.4;             % C6: left oceanic lithospheric mantle
            c6_length(ct(6))=X(j);  % x
            c6_width(ct(6))=Y(i);   % y
            c6_coord=[c6_length;c6_width]; %
            ct(6)=ct(6)+1;
        elseif (Y(i)>=(max(Y)-th_WZl) &...
                Y(i)<=max(Y) &...
                X(j)<=((Y(i)-b1)/k1+len_WZl) &...
                X(j)>=(Y(i)-b1)/k1)
            C(i,j)=0.15;               % C7: left weak zone
            c7_length(ct(7))=X(j);  % x
            c7_width(ct(7))=Y(i);   % y
            c7_coord=[c7_length;c7_width]; %
            ct(7)=ct(7)+1;
        elseif (Y(i)>(max(Y)-th_CUCm) & ...
                Y(i)<=max(Y) & ...
                X(j)>((Y(i)-b1)/k1+len_WZl) &...
                X(j)<(len_CPl+len_OPl+len_WZl+len_CPm))
            C(i,j)=0.9;             % c8, middle upper crust
            c8_length(ct(8))=X(j);  % x
            c8_width(ct(8))=Y(i);   % y
            c8_coord=[c8_length;c8_width]; %
            ct(8)=ct(8)+1;
        elseif (Y(i)>(max(Y)-th_CUCm-th_CLCm) & ...
                Y(i)<=(max(Y)-th_CUCm) & ...
                X(j)>((Y(i)-b1)/k1+len_WZl) &...
                X(j)<(len_CPl+len_OPl+len_WZl+len_CPm))
            C(i,j)=0.8;             % C9, middle lower crust
            c9_length(ct(9))=X(j);    % x
            c9_width(ct(9))=Y(i);     % y 
            c9_coord=[c9_length;c9_width]; %
            ct(9)=ct(9)+1;
        elseif (Y(i)>=(max(Y)-th_CUCm-th_CLCm-th_CLMm) &...
                Y(i)<=(max(Y)-th_CUCm-th_CLCm) & ...
                X(j)>((Y(i)-b1)/k1+len_WZl) &...
                X(j)<(len_CPl+len_OPl+len_WZl+len_CPm))
            C(i,j)=0.7;               % C10, middle lithospheric mantle
            c10_length(ct(10))=X(j);  % x
            c10_width(ct(10))=Y(i);   % y
            c10_coord=[c10_length;c10_width]; %
            ct(10)=ct(10)+1;
        elseif (Y(i)>(max(Y)-th_sedSS) & ...
                Y(i)<=(max(Y)) & ...
                X(j)>=(len_CPl+len_OPl+len_WZl+len_CPm) & ...
                X(j)<(len_CPl+len_OPl+len_WZl+len_CPm+len_OSS_CSS))
            C(i,j)=0.5;               % C11: South Sea sediment/upper crust
            c11_length(ct(11))=X(j);  % x
            c11_width(ct(11))=Y(i);   % y
            c11_coord=[c11_length;c11_width]; %
            ct(11)=ct(11)+1;
        elseif (Y(i)>(max(Y)-th_sedSS-th_OCSS) & ...
                Y(i)<=(max(Y)-th_sedSS) & ...
                X(j)>=(len_CPl+len_OPl+len_WZl+len_CPm) & ...
                X(j)<(len_CPl+len_OPl+len_WZl+len_CPm+len_OSS_CSS))
            C(i,j)=0.4;                 % C12: South Sea oceanic crust/lower crust
            c12_length(ct(12))=X(j);    % x
            c12_width(ct(12))=Y(i);     % y 
            c12_coord=[c12_length;c12_width]; %
            ct(12)=ct(12)+1;
        elseif (Y(i)>=(max(Y)-th_sedSS-th_OCSS-th_OLMSS) & ...
                Y(i)<=(max(Y)-th_sedSS-th_OCSS) & ...
                X(j)>=(len_CPl+len_OPl+len_WZl+len_CPm) & ...
                X(j)<(len_CPl+len_OPl+len_WZl+len_CPm+len_OSS_CSS))
            C(i,j)=0.3;               % C13: South Sea lithospheric mantle
            c13_length(ct(13))=X(j);  % x
            c13_width(ct(13))=Y(i);   % y
            c13_coord=[c13_length;c13_width]; %
            ct(13)=ct(13)+1;
        elseif (Y(i)>=(max(Y)-th_WZSS) & ...
                Y(i)<=max(Y) & ...
                X(j)>=(len_CPl+len_OPl+len_WZl+len_CPm+len_OSS_CSS) &...
                X(j)<=(len_CPl+len_OPl+len_WZl+len_CPm+len_OSS_CSS+len_WZm))
            C(i,j)=0.9;               % C14: middle vertical weak zone
            c14_length(ct(14))=X(j);  % x
            c14_width(ct(14))=Y(i);   % y
            c14_coord=[c14_length;c14_width]; %
            ct(14)=ct(14)+1;
        elseif (Y(i)>(max(Y)-th_sedr) & ...
                Y(i)<=(max(Y)) & ...
                X(j)<(Y(i)-b2)/k2 &...
                X(j)>(len_CPl+len_OPl+len_WZl+len_CPm+len_OSS_CSS+len_WZm))
            C(i,j)=0.2;               % C15: right sediment
            c15_length(ct(15))=X(j);  % x
            c15_width(ct(15))=Y(i);   % y
            c15_coord=[c15_length;c15_width]; %
            ct(15)=ct(15)+1;
        elseif (Y(i)>(max(Y)-th_sedr-th_OCr) & ...
                Y(i)<=(max(Y)-th_sedr) & ...
                X(j)<(Y(i)-b2)/k2 &...
                X(j)>(len_CPl+len_OPl+len_WZl+len_CPm+len_OSS_CSS+len_WZm))
            C(i,j)=0.5;                 % C16:right OC
            c16_length(ct(16))=X(j);    % x
            c16_width(ct(16))=Y(i);     % y 
            c16_coord=[c16_length;c16_width]; %
            ct(16)=ct(16)+1;
        elseif (Y(i)>=(max(Y)-th_sedr-th_OCr-th_OLMr) &...
                Y(i)<=(max(Y)-th_sedr-th_OCr) & ...
                X(j)<((Y(i)-b2)/k2) &...
                X(j)>(len_CPl+len_OPl+len_WZl+len_CPm+len_OSS_CSS+len_WZm))
            C(i,j)=0.7;                 % C17:right OLM
            c17_length(ct(17))=X(j);  % x
            c17_width(ct(17))=Y(i);   % y
            c17_coord=[c17_length;c17_width]; %
            ct(17)=ct(17)+1;
        elseif (Y(i)>=(max(Y)-th_WZr) &...
                Y(i)<=max(Y) &...
                X(j)<=((Y(i)-b2)/k2+len_WZr) &...
                X(j)>=(Y(i)-b2)/k2)
            C(i,j)=0.8;               % C18: right weak zone
            c18_length(ct(18))=X(j);  % x
            c18_width(ct(18))=Y(i);   % y
            c18_coord=[c18_length;c18_width]; %
            ct(18)=ct(18)+1;
         elseif (Y(i)>(max(Y)-th_sedrr) & ...
                Y(i)<=max(Y) & ...
                X(j)>((Y(i)-b2)/k2+len_WZr) &...
                X(j)<= max(X))
            C(i,j)=0.9;               % C19, right right sediment
            c19_length(ct(19))=X(j);  % x
            c19_width(ct(19))=Y(i);   % y
            c19_coord=[c19_length;c19_width]; %
            ct(19)=ct(19)+1;
         elseif (Y(i)>(max(Y)-th_sedrr-th_OCrr) & ...
                Y(i)<=(max(Y)-th_sedrr) & ...
                X(j)>((Y(i)-b2)/k2+len_WZr) &...
                X(j)<= max(X))
            C(i,j)=0.1;                 % C20, right right oceanic crust
            c20_length(ct(20))=X(j);    % x
            c20_width(ct(20))=Y(i);     % y 
            c20_coord=[c20_length;c20_width]; %
            ct(20)=ct(20)+1;
         elseif (Y(i)>=(max(Y)-th_sedrr-th_OCrr-th_OLMrr) &...
                Y(i)<=(max(Y)-th_sedrr-th_OCrr) & ...
                X(j)>((Y(i)-b2)/k2+len_WZr) &...
                X(j)<= max(X))
            C(i,j)=0.2;               % C21, right right OLM
            c21_length(ct(21))=X(j);  % x
            c21_width(ct(21))=Y(i);   % y
            c21_coord=[c21_length;c21_width]; %
            ct(21)=ct(21)+1;
        else                          % C22: mantle
            c22_length(ct(22))=X(j);  % x
            c22_width(ct(22))=Y(i);   % y
            c22_coord=[c22_length;c22_width]; %
            ct(22)=ct(22)+1;
        end 
        
          %% cf for strain weakening
         if (Y(i)>=(max(Y)-th_CUCl-th_CLCl-th_CLMl) & ...
                Y(i)<(max(Y)) & ...
                X(j)>=0 & ...         % C23:strain weakening cf (upper 120km)
                X(j)<=max(X))
            C2(i,j)=0.5;              
            c23_length(ct2(1))=X(j);  % x
            c23_width(ct2(1))=Y(i);   % y
            c23_coord=[c23_length;c23_width]; %
            ct2(1)=ct2(1)+1;
         else                          % C24: without strain weakening
            c24_length(ct2(2))=X(j);  % x
            c24_width(ct2(2))=Y(i);   % y
            c24_coord=[c24_length;c24_width]; %
            ct2(2)=ct2(2)+1;
         end
        
    end
end
subplot(211),pcolor(X,Y,C)  % plot in different colors
subplot(212),pcolor(X,Y,C2) % plot in different colors
hold on;
axis equal;
%----save compositional fields data---------
%----add #2---------
% Here is the example of 5 compositional fields.
% If you want to add another n,you should :
% 1)add CFi(i=1,2,...,n),
%   CFi = zeros(1,length(Y)*length(X));
% 2)add formula to evalute CFi(i=1,2,...,n),
%   CFi=[zeros(1,(ct(1)-1)+(ct(2)-1)+...+(ct(i-1)-1)),...
%        ones(1,ct(i)-1),...
%        zeros(1,length(X)*length(Y)-(ct(1)-1)-(ct(2)-1)-...-(ct(i)-1))];
% 3)add new CFi to CFsum
%
% Each Compositional Field:CFi(i=1,2,...,CFnum,exclud mantle);
CF1 = zeros(1,length(Y)*length(X));   
CF2 = zeros(1,length(Y)*length(X));   
CF3 = zeros(1,length(Y)*length(X));   
CF4 = zeros(1,length(Y)*length(X));   
CF5 = zeros(1,length(Y)*length(X)); 
CF6 = zeros(1,length(Y)*length(X));
CF7 = zeros(1,length(Y)*length(X)); 
CF8 = zeros(1,length(Y)*length(X)); 
CF9 = zeros(1,length(Y)*length(X)); 
CF10 = zeros(1,length(Y)*length(X));
CF11 = zeros(1,length(Y)*length(X));
CF12 = zeros(1,length(Y)*length(X));
CF13 = zeros(1,length(Y)*length(X));
CF14 = zeros(1,length(Y)*length(X));
CF15 = zeros(1,length(Y)*length(X));
CF16 = zeros(1,length(Y)*length(X));
CF17 = zeros(1,length(Y)*length(X));
CF18 = zeros(1,length(Y)*length(X));
CF19 = zeros(1,length(Y)*length(X));
CF20 = zeros(1,length(Y)*length(X));
CF21 = zeros(1,length(Y)*length(X));
CF22 = zeros(1,length(Y)*length(X));
CF23 = zeros(1,length(Y)*length(X));
% 
pcord=[c1_coord,c2_coord,c3_coord,c4_coord,c5_coord,c6_coord,c7_coord,...
       c8_coord,c9_coord,c10_coord,c11_coord,c12_coord,c13_coord,...
       c14_coord,c15_coord,c16_coord,c17_coord,c18_coord,c19_coord,...
       c20_coord,c21_coord,c22_coord];
%===================
pcord_SW=[c23_coord,c24_coord]; %strain weakening
%===================
% exclude mantle
CF1=[ones(1,ct(1)-1),...
     zeros(1,length(X)*length(Y)-(ct(1)-1))]; % C1
CF2=[zeros(1,ct(1)-1),ones(1,ct(2)-1),...
     zeros(1,length(X)*length(Y)-(ct(1)-1)-(ct(2)-1))]; % C2
CF3=[zeros(1,(ct(1)-1)+(ct(2)-1)),ones(1,ct(3)-1),...
     zeros(1,length(X)*length(Y)-(ct(1)-1)-(ct(2)-1)-(ct(3)-1))]; % C3
CF4=[zeros(1,(ct(1)-1)+(ct(2)-1)+(ct(3)-1)),ones(1,ct(4)-1),...
     zeros(1,length(X)*length(Y)-(ct(1)-1)-(ct(2)-1)-(ct(3)-1)-(ct(4)-1))]; % C4
CF5=[zeros(1,(ct(1)-1)+(ct(2)-1)+(ct(3)-1)+(ct(4)-1)),ones(1,ct(5)-1),...
     zeros(1,length(X)*length(Y)-(ct(1)-1)-(ct(2)-1)-(ct(3)-1)-(ct(4)-1)-...
     (ct(5)-1))]; % C5
CF6=[zeros(1,(ct(1)-1)+(ct(2)-1)+(ct(3)-1)+(ct(4)-1)+(ct(5)-1)),...
     ones(1,ct(6)-1),...
     zeros(1,length(X)*length(Y)-(ct(1)-1)-(ct(2)-1)-(ct(3)-1)-(ct(4)-1)-...
     (ct(5)-1)-(ct(6)-1))]; % C6
CF7=[zeros(1,(ct(1)-1)+(ct(2)-1)+(ct(3)-1)+(ct(4)-1)+(ct(5)-1)+(ct(6)-1)),...
     ones(1,ct(7)-1),...
     zeros(1,length(X)*length(Y)-(ct(1)-1)-(ct(2)-1)-(ct(3)-1)-(ct(4)-1)-...
     (ct(5)-1)-(ct(6)-1)-(ct(7)-1))]; % C7 
CF8=[zeros(1,(ct(1)-1)+(ct(2)-1)+(ct(3)-1)+(ct(4)-1)+(ct(5)-1)+(ct(6)-1)+...
    (ct(7)-1)),...
     ones(1,ct(8)-1),...
     zeros(1,length(X)*length(Y)-(ct(1)-1)-(ct(2)-1)-(ct(3)-1)-(ct(4)-1)-...
     (ct(5)-1)-(ct(6)-1)-(ct(7)-1)-(ct(8)-1))]; % C8
CF9=[zeros(1,(ct(1)-1)+(ct(2)-1)+(ct(3)-1)+(ct(4)-1)+(ct(5)-1)+(ct(6)-1)+...
     (ct(7)-1)+(ct(8)-1)),...
     ones(1,ct(9)-1),...
     zeros(1,length(X)*length(Y)-(ct(1)-1)-(ct(2)-1)-(ct(3)-1)-(ct(4)-1)-...
     (ct(5)-1)-(ct(6)-1)-(ct(7)-1)-(ct(8)-1)-(ct(9)-1))]; % C9
CF10=[zeros(1,(ct(1)-1)+(ct(2)-1)+(ct(3)-1)+(ct(4)-1)+(ct(5)-1)+(ct(6)-1)+...
     (ct(7)-1)+(ct(8)-1)+(ct(9)-1)),...
     ones(1,ct(10)-1),...
     zeros(1,length(X)*length(Y)-(ct(1)-1)-(ct(2)-1)-(ct(3)-1)-(ct(4)-1)-...
     (ct(5)-1)-(ct(6)-1)-(ct(7)-1)-(ct(8)-1)-(ct(9)-1)-(ct(10)-1))];%C10
CF11=[zeros(1,(ct(1)-1)+(ct(2)-1)+(ct(3)-1)+(ct(4)-1)+(ct(5)-1)+(ct(6)-1)+...
     (ct(7)-1)+(ct(8)-1)+(ct(9)-1)+(ct(10)-1)),...
     ones(1,ct(11)-1),...
     zeros(1,length(X)*length(Y)-(ct(1)-1)-(ct(2)-1)-(ct(3)-1)-(ct(4)-1)-...
     (ct(5)-1)-(ct(6)-1)-(ct(7)-1)-(ct(8)-1)-(ct(9)-1)-(ct(10)-1)-(ct(11)-1))];% C11
CF12=[zeros(1,(ct(1)-1)+(ct(2)-1)+(ct(3)-1)+(ct(4)-1)+(ct(5)-1)+(ct(6)-1)+...
     (ct(7)-1)+(ct(8)-1)+(ct(9)-1)+(ct(10)-1)+(ct(11)-1)),...
     ones(1,ct(12)-1),...
     zeros(1,length(X)*length(Y)-(ct(1)-1)-(ct(2)-1)-(ct(3)-1)-(ct(4)-1)-...
     (ct(5)-1)-(ct(6)-1)-(ct(7)-1)-(ct(8)-1)-(ct(9)-1)-(ct(10)-1)-(ct(11)-1)-...
     (ct(12)-1))];% C12
CF13=[zeros(1,(ct(1)-1)+(ct(2)-1)+(ct(3)-1)+(ct(4)-1)+(ct(5)-1)+(ct(6)-1)+...
     (ct(7)-1)+(ct(8)-1)+(ct(9)-1)+(ct(10)-1)+(ct(11)-1)+(ct(12)-1)),...
     ones(1,ct(13)-1),...
     zeros(1,length(X)*length(Y)-(ct(1)-1)-(ct(2)-1)-(ct(3)-1)-(ct(4)-1)-...
     (ct(5)-1)-(ct(6)-1)-(ct(7)-1)-(ct(8)-1)-(ct(9)-1)-(ct(10)-1)-(ct(11)-1)-...
     (ct(12)-1)-(ct(13)-1))]; % C13
CF14=[zeros(1,(ct(1)-1)+(ct(2)-1)+(ct(3)-1)+(ct(4)-1)+(ct(5)-1)+(ct(6)-1)+...
     (ct(7)-1)+(ct(8)-1)+(ct(9)-1)+(ct(10)-1)+(ct(11)-1)+(ct(12)-1)+(ct(13)-1)),...
     ones(1,ct(14)-1),...
     zeros(1,length(X)*length(Y)-(ct(1)-1)-(ct(2)-1)-(ct(3)-1)-(ct(4)-1)-...
     (ct(5)-1)-(ct(6)-1)-(ct(7)-1)-(ct(8)-1)-(ct(9)-1)-(ct(10)-1)-(ct(11)-1)-...
     (ct(12)-1)-(ct(13)-1)-(ct(14)-1))]; % C14
CF15=[zeros(1,(ct(1)-1)+(ct(2)-1)+(ct(3)-1)+(ct(4)-1)+(ct(5)-1)+(ct(6)-1)+...
    (ct(7)-1)+(ct(8)-1)+(ct(9)-1)+(ct(10)-1)+(ct(11)-1)+(ct(12)-1)+...
    (ct(13)-1)+(ct(14)-1)),...
     ones(1,ct(15)-1),...
     zeros(1,length(X)*length(Y)-(ct(1)-1)-(ct(2)-1)-(ct(3)-1)-(ct(4)-1)-...
     (ct(5)-1)-(ct(6)-1)-(ct(7)-1)-(ct(8)-1)-(ct(9)-1)-(ct(10)-1)-(ct(11)-1)-...
     (ct(12)-1)-(ct(13)-1)-(ct(14)-1)-(ct(15)-1))]; % C15
CF16=[zeros(1,(ct(1)-1)+(ct(2)-1)+(ct(3)-1)+(ct(4)-1)+(ct(5)-1)+(ct(6)-1)+...
     (ct(7)-1)+(ct(8)-1)+(ct(9)-1)+(ct(10)-1)+(ct(11)-1)+(ct(12)-1)+...
     (ct(13)-1)+(ct(14)-1)+(ct(15)-1)),...
     ones(1,ct(16)-1),...
     zeros(1,length(X)*length(Y)-(ct(1)-1)-(ct(2)-1)-(ct(3)-1)-(ct(4)-1)-...
     (ct(5)-1)-(ct(6)-1)-(ct(7)-1)-(ct(8)-1)-(ct(9)-1)-(ct(10)-1)-(ct(11)-1)-...
     (ct(12)-1)-(ct(13)-1)-(ct(14)-1)-(ct(15)-1)-(ct(16)-1))]; % C16
CF17=[zeros(1,(ct(1)-1)+(ct(2)-1)+(ct(3)-1)+(ct(4)-1)+(ct(5)-1)+(ct(6)-1)+...
     (ct(7)-1)+(ct(8)-1)+(ct(9)-1)+(ct(10)-1)+(ct(11)-1)+(ct(12)-1)+...
     (ct(13)-1)+(ct(14)-1)+(ct(15)-1)+(ct(16)-1)),...
     ones(1,ct(17)-1),...
     zeros(1,length(X)*length(Y)-(ct(1)-1)-(ct(2)-1)-(ct(3)-1)-(ct(4)-1)-...
     (ct(5)-1)-(ct(6)-1)-(ct(7)-1)-(ct(8)-1)-(ct(9)-1)-(ct(10)-1)-(ct(11)-1)-...
     (ct(12)-1)-(ct(13)-1)-(ct(14)-1)-(ct(15)-1)-(ct(16)-1)-(ct(17)-1))]; % C17
CF18=[zeros(1,(ct(1)-1)+(ct(2)-1)+(ct(3)-1)+(ct(4)-1)+(ct(5)-1)+(ct(6)-1)+...
     (ct(7)-1)+(ct(8)-1)+(ct(9)-1)+(ct(10)-1)+(ct(11)-1)+(ct(12)-1)+...
     (ct(13)-1)+(ct(14)-1)+(ct(15)-1)+(ct(16)-1)+(ct(17)-1)),...
     ones(1,ct(18)-1),...
     zeros(1,length(X)*length(Y)-(ct(1)-1)-(ct(2)-1)-(ct(3)-1)-(ct(4)-1)-...
     (ct(5)-1)-(ct(6)-1)-(ct(7)-1)-(ct(8)-1)-(ct(9)-1)-(ct(10)-1)-(ct(11)-1)-...
     (ct(12)-1)-(ct(13)-1)-(ct(14)-1)-(ct(15)-1)-(ct(16)-1)-(ct(17)-1)-...
     (ct(18)-1))]; % C18
CF19=[zeros(1,(ct(1)-1)+(ct(2)-1)+(ct(3)-1)+(ct(4)-1)+(ct(5)-1)+(ct(6)-1)+...
     (ct(7)-1)+(ct(8)-1)+(ct(9)-1)+(ct(10)-1)+(ct(11)-1)+(ct(12)-1)+...
     (ct(13)-1)+(ct(14)-1)+(ct(15)-1)+(ct(16)-1)+(ct(17)-1)+(ct(18)-1)),...
     ones(1,ct(19)-1),...
     zeros(1,length(X)*length(Y)-(ct(1)-1)-(ct(2)-1)-(ct(3)-1)-(ct(4)-1)-...
     (ct(5)-1)-(ct(6)-1)-(ct(7)-1)-(ct(8)-1)-(ct(9)-1)-(ct(10)-1)-(ct(11)-1)-...
     (ct(12)-1)-(ct(13)-1)-(ct(14)-1)-(ct(15)-1)-(ct(16)-1)-(ct(17)-1)-...
     (ct(18)-1)-(ct(19)-1))]; % C19
CF20=[zeros(1,(ct(1)-1)+(ct(2)-1)+(ct(3)-1)+(ct(4)-1)+(ct(5)-1)+(ct(6)-1)+...
     (ct(7)-1)+(ct(8)-1)+(ct(9)-1)+(ct(10)-1)+(ct(11)-1)+(ct(12)-1)+...
     (ct(13)-1)+(ct(14)-1)+(ct(15)-1)+(ct(16)-1)+(ct(17)-1)+(ct(18)-1)+...
     (ct(19)-1)),...
     ones(1,ct(20)-1),...
     zeros(1,length(X)*length(Y)-(ct(1)-1)-(ct(2)-1)-(ct(3)-1)-(ct(4)-1)-...
     (ct(5)-1)-(ct(6)-1)-(ct(7)-1)-(ct(8)-1)-(ct(9)-1)-(ct(10)-1)-(ct(11)-1)-...
     (ct(12)-1)-(ct(13)-1)-(ct(14)-1)-(ct(15)-1)-(ct(16)-1)-(ct(17)-1)-...
     (ct(18)-1)-(ct(19)-1)-(ct(20)-1))]; % C20
CF21=[zeros(1,(ct(1)-1)+(ct(2)-1)+(ct(3)-1)+(ct(4)-1)+(ct(5)-1)+(ct(6)-1)+...
     (ct(7)-1)+(ct(8)-1)+(ct(9)-1)+(ct(10)-1)+(ct(11)-1)+(ct(12)-1)+...
     (ct(13)-1)+(ct(14)-1)+(ct(15)-1)+(ct(16)-1)+(ct(17)-1)+(ct(18)-1)+...
     (ct(19)-1)+(ct(20)-1)),...
     ones(1,ct(21)-1),...
     zeros(1,length(X)*length(Y)-(ct(1)-1)-(ct(2)-1)-(ct(3)-1)-(ct(4)-1)-...
     (ct(5)-1)-(ct(6)-1)-(ct(7)-1)-(ct(8)-1)-(ct(9)-1)-(ct(10)-1)-(ct(11)-1)-...
     (ct(12)-1)-(ct(13)-1)-(ct(14)-1)-(ct(15)-1)-(ct(16)-1)-(ct(17)-1)-...
     (ct(18)-1)-(ct(19)-1)-(ct(20)-1)-(ct(21)-1))]; % C21
% ==========CF_SW -> cf for strain weakening
CF_SW=[3*ones(1,ct2(1)-1),...
     zeros(1,length(X)*length(Y)-(ct2(1)-1))]; 
%================================
 
CFsum =[CF1;CF2;CF3;CF4;CF5;CF6;CF7;CF8;CF9;CF10;CF11;CF12;CF13;CF14;CF15;CF16;...
        CF17;CF18;CF19;CF20;CF21]; % compositional fields data
Cdata =[pcord;CFsum]';         % compositional fields data with coordinates
sortDatax = sortrows(Cdata,1);
sortDatay = sortrows(sortDatax,2);
compfdata1 = sortDatay;
[x_num,y_num]=size(compfdata1);
%===================================
% sort for strain weakening
CFsum2 =[CF_SW]; % compositional fields data
Cdata2 =[pcord_SW;CFsum2]';         % compositional fields data with coordinates
sortDatax2 = sortrows(Cdata2,1);
sortDatay2 = sortrows(sortDatax2,2);
compfdata2 = sortDatay2;
cf_sw = compfdata2(:,3);
% column1 is x axis,column2 is y axis (compfdata)
compfdata=[compfdata1(:,1),compfdata1(:,2),compfdata1(:,3:y_num),cf_sw];
%===================================

% save data
StrCFnum=strcat('_',num2str(CFnum));
filename=strcat(data_dir,'composition.txt'); 
fid=fopen(filename,'w');
l1='# Test data for ascii data compositional initial conditions.';
l2='# Only next line is parsed in format: [nx] [ny] because of keyword "POINTS:"';
fprintf(fid,'%s\n',l1,l2,l3);
fclose(fid);
%dlmwrite(filename,compfdata,'-append','delimiter','\t','precision','%8.6f');
dlmwrite(filename,compfdata,'-append','delimiter','\t','precision','%6.0f');
%
%
fprintf('--Saving data--\n');
filename=strcat(data_tempdir,'C1-left continental upper crust.txt'); 
dlmwrite(filename,c1_coord,'delimiter','\t','precision','%8.6f');
%
filename=strcat(data_tempdir,'C2-left continental lower crust.txt'); 
dlmwrite(filename,c2_coord,'delimiter','\t','precision','%8.6f');
%
filename=strcat(data_tempdir,'C3-left continental lithospheric mantle.txt'); 
dlmwrite(filename,c3_coord,'delimiter','\t','precision','%8.6f');
%
filename=strcat(data_tempdir,'C4-left sediment.txt'); 
dlmwrite(filename,c4_coord,'delimiter','\t','precision','%8.6f');
%
filename=strcat(data_tempdir,'C5-left oceanic crust.txt'); 
dlmwrite(filename,c5_coord,'delimiter','\t','precision','%8.6f');
%
filename=strcat(data_tempdir,'C6-left oceanic lithospheric mantle.txt'); 
dlmwrite(filename,c6_coord,'delimiter','\t','precision','%8.6f');
%
filename=strcat(data_tempdir,'C7-left weak zone.txt'); 
dlmwrite(filename,c7_coord,'delimiter','\t','precision','%8.6f');
%
filename=strcat(data_tempdir,'C8-middle upper crust.txt'); 
dlmwrite(filename,c8_coord,'delimiter','\t','precision','%8.6f'); 
%
filename=strcat(data_tempdir,'C9-middle lower crust.txt'); 
dlmwrite(filename,c9_coord,'delimiter','\t','precision','%8.6f'); 
%
filename=strcat(data_tempdir,'C10-middle continental lithospheric mantle.txt'); 
dlmwrite(filename,c10_coord,'delimiter','\t','precision','%8.6f'); 
%
filename=strcat(data_tempdir,'C11-south sea sedment.txt'); 
dlmwrite(filename,c11_coord,'delimiter','\t','precision','%8.6f');
%
filename=strcat(data_tempdir,'C12-south sea crust.txt'); 
dlmwrite(filename,c12_coord,'delimiter','\t','precision','%8.6f');
%
filename=strcat(data_tempdir,'C13-south sea lithospheric mantle.txt'); 
dlmwrite(filename,c13_coord,'delimiter','\t','precision','%8.6f');
%% If South Sea is ocean, comment this section
%
% filename=strcat(data_tempdir,'C11-right continental upper crust.txt'); 
% dlmwrite(filename,c11_coord,'delimiter','\t','precision','%8.6f');
% %
% filename=strcat(data_tempdir,'C12-right continental lower crust.txt'); 
% dlmwrite(filename,c12_coord,'delimiter','\t','precision','%8.6f');
% %
% filename=strcat(data_tempdir,'C13-right continental lithospheric mantle.txt'); 
% dlmwrite(filename,c13_coord,'delimiter','\t','precision','%8.6f');
%
%%
filename=strcat(data_tempdir,'C14-vertical weak zone.txt'); 
dlmwrite(filename,c14_coord,'delimiter','\t','precision','%8.6f');
%
filename=strcat(data_tempdir,'C15-right sediment.txt'); 
dlmwrite(filename,c15_coord,'delimiter','\t','precision','%8.6f');
%
filename=strcat(data_tempdir,'C16-right crust.txt'); 
dlmwrite(filename,c16_coord,'delimiter','\t','precision','%8.6f');
%
filename=strcat(data_tempdir,'C17-right lithospheric mantle.txt'); 
dlmwrite(filename,c17_coord,'delimiter','\t','precision','%8.6f');
%
filename=strcat(data_tempdir,'C18-right weak zone.txt'); 
dlmwrite(filename,c18_coord,'delimiter','\t','precision','%8.6f');
%
filename=strcat(data_tempdir,'C19-right right sediment.txt'); 
dlmwrite(filename,c19_coord,'delimiter','\t','precision','%8.6f');
%
filename=strcat(data_tempdir,'C20-right right crust.txt'); 
dlmwrite(filename,c20_coord,'delimiter','\t','precision','%8.6f');
%
filename=strcat(data_tempdir,'C21-right right lithospheric mantle.txt'); 
dlmwrite(filename,c21_coord,'delimiter','\t','precision','%8.6f');
%
filename=strcat(data_tempdir,'C22-mantle.txt'); 
dlmwrite(filename,c22_coord,'delimiter','\t','precision','%8.6f');
%====================================
%=== cf for strain weakening=========
filename=strcat(data_tempdir,'C23-lithophere-SW.txt'); 
dlmwrite(filename,c23_coord,'delimiter','\t','precision','%8.6f');
%-----------------------
fprintf('Done!\n');
%
toc  % clock off
end
