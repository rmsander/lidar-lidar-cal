function [] = drawtriad(axs,c1,c2,c3,s,mat,leg)

hold on
mom = hgtransform('Parent',axs);
kids(1) = plot3([0,1],[0,0],[0,0],'Color',c1,'Tag','X-Axis','Parent',mom);
kids(2) = plot3([0,0],[0,1],[0,0],'Color',c2,'Tag','Y-Axis','Parent',mom);
kids(3) = plot3([0,0],[0,0],[0,1],'Color',c3,'Tag','Z-Axis','Parent',mom);

for j = 1:numel(kids)
    xdata = get(kids(j),'XData');
    ydata = get(kids(j),'YData');
    zdata = get(kids(j),'ZData');
    set(kids(j),'XData',xdata*s,'YData',ydata*s,'ZData',zdata*s);
end
set(kids,'LineWidth',2)
t= text(0.05, 0.05, 0.05, leg, 'Parent', mom);
set(mom,'Matrix',mat)




drawnow


end

