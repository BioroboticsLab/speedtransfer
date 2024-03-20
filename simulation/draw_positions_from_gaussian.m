function [positions] = draw_positions_from_gaussian(m, Cov)
%DRAW_POSITIONS_FROM_GAUSSIAN Summary of this function goes here
%   Detailed explanation goes here

positions = mvnrnd(m,Cov);

end

