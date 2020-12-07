function h = triad(varargin)
% TRIAD adds a reference frame to an axes using red/green/blue to represent
% the x, y, and z-directions respectively.
%   TRIAD This function adds a reference frame a specified parent or the 
%   current axes using red/green/blue to represent the x, y, and 
%   z-directions respectively. An hgtransform object is returned that is 
%   the parent of three line objects representing the specified axes. A 
%   number of parameters can be varied allowing the user to adjust the 
%   reference frame representation as needed including:
%       'linestyle' - [ {-} | -- | : | -. | none ]
%       'linewidth' - Used to define thickness lines representing axes 
%                     Default value is 0.50
%                     Value must be finite and greater than zero
%       'matrix'    - 4x4 homogenious transform
%       'parent'    - Axes or other hgtransform handle
%       'scale'     - Used to define the length of each axes line
%                     Default value is 1.00
%                     Value(s) must be finite and greater than zero
%                     A scalar value scales the x, y, and z-axis equally
%                     A 3-element array (e.g. [1,2,3]) scales each of the
%                       axes seperately.
%       'tag        - String describing object 
%
%   Example
%       axs = axes;
%       view(3);
%       daspect([1 1 1]);
%       h = triad('Parent',axs,'Scale',10,'LineWidth',3,...
%           'Tag','Triad Example','Matrix',...
%           makehgtform('xrotate',pi/4,'zrotate',pi/3,'translate',[1,2,3]));
%       H = get(h,'Matrix');
%       for theta = 0:360
%           set(h,'Matrix',H*makehgtform(...
%               'xrotate',deg2rad(theta),'zrotate',deg2rad(theta)));
%           drawnow
%       end
%
%   See also hgtransform
%
%   (c) M. Kutzer 20Oct2014, USNA

%Updates
%   19Dec2014 - Updated parent definition and extended documentation

% TODO - create a triad class

%% Find or default parent
idx = find( strcmpi('parent',varargin) );
if ~isempty(idx)
    if numel(idx) > 1
        idx = idx(end);
        warning(sprintf('Multiple Parents are specified, using %d.',idx));
    end
    mom = varargin{idx+1};
    axs = ancestor(mom,'axes','toplevel');
    hold(axs,'on');
else
    mom = gca;
    hold(mom,'on');
end

%% Create triad
h = hgtransform('Parent',mom);
kids(1) = plot3([0,1],[0,0],[0,0],'Color',[1,0,0],'Tag','X-Axis','Parent',h);
kids(2) = plot3([0,0],[0,1],[0,0],'Color',[0,1,0],'Tag','Y-Axis','Parent',h);
kids(3) = plot3([0,0],[0,0],[0,1],'Color',[0,0,1],'Tag','Z-Axis','Parent',h);

%% Update properties
for i = 1:2:numel(varargin)
    switch lower(varargin{i})
        case 'linestyle'
            set(kids,varargin{i},varargin{i+1});
        case 'linewidth'
            set(kids,varargin{i},varargin{i+1});
        case 'matrix'
            set(h,varargin{i},varargin{i+1});
        case 'parent'
            %do nothing, property handled earlier
            %set(h,varargin{i},varargin{i+1});
            %daspect([1 1 1]);
        case 'scale'
            s = varargin{i+1};
            if numel(s) == 1
                s = repmat(s,1,3);
            end
            if numel(s) ~= 3
                error('The scaling factor must be a singular value or a 3-element array.');
            end
            for j = 1:numel(kids)
                xdata = get(kids(j),'XData');
                ydata = get(kids(j),'YData');
                zdata = get(kids(j),'ZData');
                set(kids(j),'XData',xdata*s(1),'YData',ydata*s(2),'ZData',zdata*s(3));
            end
        case 'tag'
            set(h,varargin{i},varargin{i+1});
        otherwise
            % TODO - add check for properties in line or hgtransform, and
            % update property accordingly.
            warning(sprintf('Ignoring "%s," unexpected property.',varargin{i}));
    end
end