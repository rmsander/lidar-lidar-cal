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

H_vel2f=[6.322853040923779311e-01 7.065021776617672611e-01 3.179150313967388231e-01 2.736453979821516433e-02;...
-6.309036382479351612e-01 7.077107086327081298e-01 -3.179719360762964797e-01 -1.552742243901841269e-02;...
-4.496397374280075021e-01 4.752323329342278413e-04 8.932098749341624844e-01 1.027682430031146565e-01;...
0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00];


H_vel2r=[-6.580254646951748665e-01 -6.902288734836319328e-01 -3.009760622079694081e-01 -1.633659613214694384e-01;...
6.263459395925998763e-01 -7.235873770229015456e-01 2.900208126479554283e-01 1.718226535666872445e-01;...
-4.179632182005519891e-01 2.325945535411738652e-03 9.084609723091048306e-01 2.500343229522681079e-02;...
0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00];

%%  Plot initial guesses
figure;
axs = axes;
view(3);
daspect([1 1 1]);
s=1/10;
hold on
grid on

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