clear all
clc

base= [eul2rotm([0,-0.0049916,0.7827801355]),[0.038; 0; 0.405];[0 0 0 1]]
fronti= [eul2rotm([0 0.445058959 -0.005236]),[0.138; -0.01; 0.285];[0 0 0 1]]
reari= [eul2rotm([0 0.428 3.15555525359]),[-0.47; 0; 0.21];[0 0 0 1]]




%%
fig = figure('Name','Results Visualization','units','normalized','outerposition',[0 0 1 1]);
axs = axes('Parent',fig);
%axs = axes;
view(3);
daspect([1 1 1]);
h = triad('Parent',axs,'Scale',1/10,'LineWidth',3,...
    'Tag','Triad Example','Matrix',...
    base);
H = get(h,'Matrix');
set(h,'Matrix',H);
drawnow
hold on

fi = triad('Parent',axs,'Scale',1/10,'LineWidth',3,...
    'Tag','Triad Example','Matrix',...
    fronti);
Fi = get(fi,'Matrix');
set(fi,'Matrix',Fi);
drawnow

ri = triad('Parent',axs,'Scale',1/10,'LineWidth',3,...
    'Tag','Triad Example','Matrix',...
    reari);
Ri = get(ri,'Matrix');
set(ri,'Matrix',Ri);
drawnow

%%

H_v2f=[6.343942765163238517e-01 7.089467095487633763e-01 3.081208609349767036e-01 6.324324189026664378e-02;...
-6.398928392968971091e-01 7.052531742994365693e-01 -3.052132276902886909e-01 -7.761486928506447502e-02;...
-4.336831287240552890e-01 -3.538807786493930897e-03 9.010585001538827798e-01 -1.203171825678759355e-01;...
0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00]

f = triad('Parent',axs,'Scale',1/10,'LineWidth',3,'linestyle',':',...
    'Tag','Triad Example','Matrix',...
    base*H_v2f);
F = get(f,'Matrix');
set(f,'Matrix',F);
drawnow

H_v2r=[-6.559655232110801482e-01 -6.952771863905571337e-01 -2.937666870889967252e-01 -3.611183063980281105e-01;...
6.325694466803306604e-01 -7.187333227268444258e-01 2.885798085947287039e-01 3.582685924496138430e-01;...
-4.117828644868002930e-01 3.470574427994095167e-03 9.112753851762842583e-01 -1.931998562328461067e-01;...
0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00]


r = triad('Parent',axs,'Scale',1/10,'LineWidth',3,'linestyle',':',...
    'Tag','Triad Example','Matrix',...
    base*H_v2r);
R = get(r,'Matrix');
set(r,'Matrix',R);
drawnow

