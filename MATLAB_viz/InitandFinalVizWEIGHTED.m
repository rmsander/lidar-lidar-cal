clear all
clc

vel=[ 1.00000000e+00 -5.89728607e-17 -1.15231776e-19  0.00000000e+00;...
 -5.89728607e-17  1.00000000e+00 -6.16213052e-20  0.00000000e+00;...
 -1.15231776e-19 -6.16213052e-20  1.00000000e+00  0.00000000e+00;...
  0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00];

H_vel2fi=[ 0.63439428  0.70894671  0.30812086  0.06324324;...
 -0.63989284  0.70525317 -0.30521323 -0.07761487;...
 -0.43368313 -0.00353881  0.9010585  -0.12031718;...
  0.          0.          0.          1.        ];

H_vel2ri=[-0.65596552 -0.69527719 -0.29376669 -0.36111831;...
  0.63256945 -0.71873332  0.28857981  0.35826859;...
 -0.41178286  0.00347057  0.91127539 -0.19319986;...
  0.          0.          0.          1.        ];

H_vel2f=[6.340263240164707437e-01 7.065047908381872910e-01 3.144226470482849312e-01 2.744631626888732398e-02;...
-6.309488247000096806e-01 7.077011390706251515e-01 -3.179035677179736785e-01 -1.554647028772772159e-02;...
-4.471176590829727959e-01 3.174630817766699309e-03 8.944695191314980809e-01 1.029086134551797371e-01;...
0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00];

H_vel2r=[-6.597561821640627322e-01 -6.888474965256450311e-01 -3.003513053520014964e-01 -1.635326692153628025e-01;...
6.265123809243304809e-01 -7.249059059870481825e-01 2.863453579397147730e-01 1.720175158173550822e-01;...
-4.149747180790974710e-01 7.443087048951690932e-04 9.098326381042424194e-01 2.536711932197747640e-02;...
0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00];

%%  Plot initial guesses
figure;
axs = axes;
view(3);
daspect([1 1 1]);
s=1/10;
hold on

%Plot vel
c1=[1,0,0];
c2=[0,1,0];
c3=[0,0,1];
mat=vel;
leg='main';
drawtriad(axs,c1,c2,c3,s,mat,leg)

%Plot fi
c1=[1,0,0];
c2=[0,1,0];
c3=[0,0,1];
mat=vel*H_vel2fi;
leg='front';
drawtriad(axs,c1,c2,c3,s,mat,leg)


%Plot ri
c1=[1,0,0];
c2=[0,1,0];
c3=[0,0,1];
mat=vel*H_vel2ri;
leg='rear';
drawtriad(axs,c1,c2,c3,s,mat,leg)

xlabel('X (m)')
ylabel('Y (m)')
zlabel('Z (m)')

xlim([-0.5 0.2])

%% Plot final guesses
figure;
axs = axes;
grid on
view(3);
daspect([1 1 1]);
s=1/10;
hold on

%Plot vel
c1=[1,0,0];
c2=[0,1,0];
c3=[0,0,1];
mat=vel;
leg='main';
drawtriad(axs,c1,c2,c3,s,mat,leg)

%Plot fi
c1=[1,0,0];
c2=[0,1,0];
c3=[0,0,1];
mat=vel*H_vel2fi;
leg='front';
drawtriad(axs,c1,c2,c3,s,mat,leg)


%Plot ri
c1=[1,0,0];
c2=[0,1,0];
c3=[0,0,1];
mat=vel*H_vel2ri;
leg='rear';
drawtriad(axs,c1,c2,c3,s,mat,leg)


%Plot f
c1=[0 1 1];
c2=[1 1 0];
c3=[0 0 0];
mat=vel*H_vel2f;
leg='front(final)';
drawtriad(axs,c1,c2,c3,s,mat,leg)

%Plot r
c1=[0 1 1];
c2=[1 1 0];
c3=[0 0 0];
mat=vel*H_vel2r;
leg='rear(final)';
drawtriad(axs,c1,c2,c3,s,mat,leg)


xlabel('X (m)')
ylabel('Y (m)')
zlabel('Z (m)')

xlim([-0.5 0.2])


% %% Plot f close up
% 
% figure;
% axs = axes;
% view(3);
% daspect([1 1 1]);
% s=1/10000;
% hold on
% 
% %Plot fi
% c1=[1,0,0];
% c2=[0,1,0];
% c3=[0,0,1];
% mat=vel*H_vel2fi;
% leg='front';
% drawtriad(axs,c1,c2,c3,s,mat,leg)
% 
% %Plot f
% c1=[0 1 1];
% c2=[1 1 0];
% c3=[0 0 0];
% mat=vel*H_vel2f;
% leg='';
% drawtriad(axs,c1,c2,c3,s,mat,leg)
% 
% %% Plot r close up
% 
% figure;
% axs = axes;
% view(3);
% daspect([1 1 1]);
% s=1/10000;
% hold on
% 
% %Plot fi
% c1=[1,0,0];
% c2=[0,1,0];
% c3=[0,0,1];
% mat=vel*H_vel2ri;
% leg='rear';
% drawtriad(axs,c1,c2,c3,s,mat,leg)
% 
% %Plot f
% c1=[0 1 1];
% c2=[1 1 0];
% c3=[0 0 0];
% mat=vel*H_vel2r;
% leg='';
% drawtriad(axs,c1,c2,c3,s,mat,leg)