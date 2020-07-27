function [ LeftWheelVelocity, RightWheelVelocity ] = calculateWheelSpeeds( vu, omega, parameters )
%CALCULATEWHEELSPEEDS This function computes the motor velocities for a differential driven robot

wheelRadius = parameters.wheelRadius;
halfWheelbase = parameters.interWheelDistance/2;

A=[wheelRadius/2  wheelRadius/2;wheelRadius/(2*halfWheelbase)  -wheelRadius/(2*halfWheelbase)];
A=inv(A)*[vu;omega];
LeftWheelVelocity = A(2,1);
RightWheelVelocity =A(1,1);
end
