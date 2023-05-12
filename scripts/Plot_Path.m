clear;clc;
close all;

% Data Load
obs = load('envlog.txt');   % envlog:  bar-x, bar-y
state = load('testlog.txt');    % testlog: pos-x, pos-y, vel-x, vel-y,

% Setting
uav_alt = 3;                                % UAV Alt
obs_radiu = 0.3;                            % obstacles radiu
mcm = [0.41 0.41 0.41; 0 1 0; 0 0 1];       % obs color
floor_color = [0.8, 0.8, 0.8];              % floor color
map_size = [-0.1,12, -0.1,12, 0, 5];        % map size
path_color_1 = [0.95686, 0.6431, 0.37647];  % path color

obs_x = obs(:,1);
obs_y = obs(:,2);
uav_x = state(:,1);
uav_y = state(:,2);
uav_z = (0*(1:length(uav_x)) + 1)*uav_alt;

for i = 1:100
    if(i > 1 && obs_x(i) == obs_x(1) && obs_y(i) == obs_y(1))
        break;
    else
        cobs_x(i) = obs_x(i);
        cobs_y(i) = obs_y(i);
    end
end
[x,y,z] = cylinder(obs_radiu);    % Éú³ÉÔ²Öù

figure()
% Plot obstacles
colormap(mcm);
for i = 1:length(cobs_x)
    z(2,:) = rand(1)*2 + uav_alt;
    surf(x + cobs_x(i),y + cobs_y(i),z,'FaceColor','r');
    % shading(gca,'interp')
    % shading faceted
    shading flat
    daspect([1,1,1])
    hold on
end
% Plot floor
floor_x = [-100 100 100 -100];
floor_y = [-100 -100 100 100];
patch(floor_x, floor_y, floor_color); % light gray
hold on;
% Plot Start and End point
plot3(uav_x(1), uav_y(1), uav_z(1),'r*','Linewidth',8);
hold on;
plot3(uav_x(end), uav_y(end), uav_z(end),'y*','Linewidth',12);
hold on;
% Plot Path
plot3(uav_x, uav_y, uav_z,'Color',path_color_1,'Linewidth',3);
axis(map_size);grid off;


