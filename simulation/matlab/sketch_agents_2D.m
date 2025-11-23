% INITIALIZE AGENTS
% center agents on 
mu = [0 0];

% covariance of agent distribution
Cov = eye(2); 

% number of agents to sample
N = 100; 

% draw 2D positions 
agents = mgd(100, 2, mu, Cov);

speed_xy_avg    = 1;
speed_xy_std    = 1;

window_size = 300;

% SIMULATE RANDOM WALK 

for t = 1:1000
    speeds          = speed_xy_std*randn(N,1) + speed_xy_avg;

    velocities      = speeds .* [sin(agents(:,3)) cos(agents(:,3))];

    % update positions
    agents(:,1:2)   = agents(:,1:2) + velocities;

    speeds_trn      = speed_trn_std*randn(N,1) + speed_trn_avg;

    % update orientations
    agents(:,3)     = agents(:,3) + speeds_trn;
    
    plot(agents(:,1), agents(:,2), '.')
    
    axis([-window_size window_size -window_size window_size])
    %pause(.1)
    drawnow
end

 plot(agents(:,1), agents(:,2), '.')