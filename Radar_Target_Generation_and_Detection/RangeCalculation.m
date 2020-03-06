clc
clear
close all

% TODO : Find the Bsweep of chirp for 1 m resolution
c = 3 *10^8;                         %speed of light in meter/sec
delta_r = 1;                          %range resolution in meters
Bsweep = c/2*delta_r;        %Bsweep calculation

% TODO : Calculate the chirp time based on the Radar's Max Range
range_max = 300;                         %given radar maximum range
Ts = 5.5 * 2 * range_max/c;          %5.5 times for the trip time for maximum range

% TODO : define the frequency shifts 
beat_freq = [0, 1.1e6, 13e6, 24e6];       %given beat frequency or all the targets
calculated_range = c * Ts * beat_freq / (2*Bsweep);

% Display the calculated range
disp(calculated_range);