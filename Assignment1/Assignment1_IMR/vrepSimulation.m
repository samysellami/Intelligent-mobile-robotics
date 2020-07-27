%% V-REP Simulation Exercise 3: Kinematic Control
% Tests the implemented control algorithm within a V-Rep simulation.

% In order to run the simulation:
%   - Start V-Rep
%   - Load the scene matlab/common/vrep/mooc_exercise.ttt
%   - Hit the run button
%   - Start this script

% Parameters setup
 
% Define parameters for Dijkstra and Dynamic Window Approach
parameters.dist_threshold= 0.25; % threshold distance to goal
parameters.angle_threshold = 2; % threshold orientation to goal

% Initialize connection with V-Rep
startup;
connection = simulation_setup();
connection = simulation_openConnection(connection, 0);
simulation_start(connection);

% Get static data from V-Rep
bob_init(connection);

parameters.wheelDiameter = bob_getWheelDiameter(connection);
parameters.wheelRadius = parameters.wheelDiameter/2.0;
parameters.interWheelDistance = bob_getInterWheelDistance(connection);
parameters.scannerPoseWrtBob = bob_getScannerPose(connection);

% controller parameters
parameters.Krho = 0.5;
parameters.Kalpha = 1.5;
parameters.Kbeta = -0.6;
parameters.backwardAllowed = false;
parameters.useConstantSpeed = true;
parameters.constantSpeed = 0.2;

%bob_setTargetGhostPose(connection, -1, 0, 0);
bob_setTargetGhostVisible(connection, 1);


% CONTROL LOOP.


% CONTROL STEP.
% Get pose and goalPose from vrep
[x, y, theta] = bob_getPose(connection);
[xg, yg, thetag] = bob_getTargetGhostPose(connection);
    
%calculate the shortest path fromo source to destination
source.row= -round((512*y)/5)+256;
source.col= round((512*x)/5)+256;
destination.row= -round((512*yg)/5)+256;
destination.col= round((512*xg)/5)+256;
%    
map= bob_getMap(connection);
[Path,Path_dist]= calculatePath( map, source, destination );
    

%% CONTROL STEP.
EndCond = 0;
len=length(Path_dist);  
n=len;
step=floor(len/n);

[x, y, theta] = bob_getPose(connection);
xg=Path_dist(step,1);
yg=Path_dist(step,2);
lambda = atan2(yg-y, xg-x);     % angle of the vector pointing from the robot to the goal in the inertial frame
alpha = lambda - theta;         % angle of the vector pointing from the robot to the goal in the robot frame
alpha = normalizeAngle(alpha);

if (alpha>pi/2 || alpha <-pi/2)
    %parameters.backwardAllowed=true
end 

for ind=step:step:len
    if (len-ind)<10
        parameters.dist_threshold=0.25;
        parameters.angle_threshold = 0.1
    end
    xg=Path_dist(ind,1);
    yg=Path_dist(ind,2);
    thetag=normalizeAngle(atan2((yg-y),xg-x));
    while (~EndCond)
        % Get pose and goalPose from vrep
        [x, y, theta] = bob_getPose(connection);
        %[xg, yg, thetag] = bob_getTargetGhostPose(connection);
        
        % run control step
        [ vu, omega ] = calculateControlOutput([x, y, theta], [xg, yg, thetag], parameters);

        % Calculate wheel speeds
        [LeftWheelVelocity, RightWheelVelocity ] = calculateWheelSpeeds(vu, omega, parameters);

        % End condition
        dtheta = abs(normalizeAngle(theta-thetag));

        rho = sqrt((xg-x)^2+(yg-y)^ 2); % pythagoras theorem, sqrt(dx^2 + dy^2)
        %EndCond = (rho < parameters.dist_threshold) || rho > 5;
        EndCond = (rho < parameters.dist_threshold && dtheta < parameters.angle_threshold) || rho > 5;    
    
        % SET ROBOT WHEEL SPEEDS.
        bob_setWheelSpeeds(connection, LeftWheelVelocity, RightWheelVelocity);
    end
    EndCond=0;
end

% Bring Bob to standstill
bob_setWheelSpeeds(connection, 0.0, 0.0);

simulation_stop(connection);
simulation_closeConnection(connection);

% msgbox('Simulation ended');
