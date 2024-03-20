
% INITIALIZE AGENTS
% center agents on 
mu1 = [-1 0];
mu2 = [1 0];

% covariance of agent distribution
Cov = eye(2); 

% number of agents to sample
N1  = 100; 
N2  = 100;
N   = N1+N2;

idx_blue = 1:N1;
idx_green = N1+1:N;

% hive bounds [XMIN XMAX YMIN YMAX]
bounds = 2*[-1 1 -1 1]

% draw 2D positions and angle between 0..2pi
agents = [mgd(N1, 2, mu1, Cov) 2*pi*rand(N1,1)];
agents = [agents; mgd(N2, 2, mu2, Cov) 2*pi*rand(N2,1)];

% we add the motion vector pointing home, scaled by this parameter
homing_pull = .005;

speed_xy_avg    = 1;
speed_xy_std    = 1;

speed_trn_avg   = 0;
speed_trn_std   = 0.5; 

threshold_dist = .15;

window_size = 4;

% SIMULATE RANDOM WALK 

speed_transfers = zeros(N, 1); 

day_duration = 200;
sim_duration = 1000;

OUT_speeds  = nan(N,sim_duration);
OUT_cov     = nan(2, sim_duration);

fig1 = figure(1);
fig2 = figure(2);
fig3 = figure(3);

% collect aggregated interactions
B = zeros(N,N);

for t = 1:sim_duration
    
    last_agents = agents;
    
    %% external stimuli (e.g. related to weather variables) ...
    % ... are periodic drivers of the motion speed of group 1, one period every <day_duration> time steps
    speed_external_driver = 0.5*(sin(t*2*pi/day_duration)+1)*[ones(N1, 1); zeros(N2, 1)];    
        
    %% update random motions and homing 
    % draw motion speeds from gaussian, clip below 0, add speed transfers
    speeds          = max(0,speed_xy_std*randn(N,1) + speed_xy_avg) + speed_transfers + speed_external_driver;

    
    % store for analysis
    OUT_speeds(:,t) = speeds;
    
    % walk into direction agents are facing, scale by speed
    velocities      = speeds .* [sin(agents(:,3)) cos(agents(:,3))];
    
    % correct velocities
    home1           = repmat(mu1, N1, 1) - agents(idx_blue,1:2);
    home1           = normalize(home1, 2, 'norm');
    velocities(1:N1, :) = velocities(idx_blue, :) + homing_pull*norm(home1)^3*(home1-velocities(idx_blue, :));
    
    home2           = repmat(mu2, N2, 1) - agents(idx_green,1:2);
    home2           = normalize(home2, 2, 'norm');
    velocities(N1+1:end, :) = velocities(idx_green, :) + homing_pull*norm(home2)^3*(home2-velocities(idx_green, :));
    
    % update positions
    agents(:,1:2)   = agents(:,1:2) + velocities;
    
    % correct velocities to have agents remain in hive
    % find agents out-of-bounds, set them back on the bounds
    agents( find(agents(:,1) < bounds(1)), 1 ) = bounds(1);
    agents( find(agents(:,1) > bounds(2)), 1 ) = bounds(2);
    agents( find(agents(:,2) < bounds(3)), 2 ) = bounds(3);
    agents( find(agents(:,2) > bounds(4)), 2 ) = bounds(4);
    
    % extract speeds for post-sim analysis
    % OUT_speeds
    

    % draw random rotation speed changes
    speeds_trn      = speed_trn_std*randn(N,1) + speed_trn_avg;

    % update orientations
    agents(:,3)     = agents(:,3) + speeds_trn;
        
    %% identify interactions for speed transfer
    % matrix of agent positions repeated along dim 2
    % [x1 y1 x1 y1 x1 y1 ... ; 
    %  x2 y2 x2 y2 x2 y2 ... ; ...]
    % size: N x 2N
    Q2 = repmat(agents(:,1:2), 1, N);
    
    % matrix with agent positions repeated along dim 1
    % [x1 y1 x2 y2 x3 y3 ...; 
    %  x1 y1 x2 y2 x3 y3 ...; 
    %  x1 y1 x2 y2 x3 y3 ...; 
    %  x1 y1 x2 y2 x3 y3 ...; 
    % ...]
    % size: N x 2N
    Q1 = repmat( reshape(agents(:,1:2)', 2*N, 1)' , N, 1);
    
    % squared differences  between all agents' positions
    % [dx11 dy11 dx12 dy12 dx13 dy13 ...;
    %  dx21 dy21 dx22 dy22 dx23 dy23 ...;
    %  ...]
    % size: N x 2N
    Q = (Q2-Q1).^2;
    
    
    % actual distances 
    % add 2*threshold distance to have self-distance above threshold
    % size: N x N
    D = (2*threshold_dist)*eye(N)+sqrt(Q(:,1:2:end) + Q(:,2:2:end));
    
    % binary interaction matrix
    W = D < threshold_dist;
    
    % these ones are sufficiently close, i.e. they interact
    [i,j] = find(W);
    
    B = B + W;
    
    % transfer falls off every time step
    speed_transfers = speed_transfers / 2;
    %speed_transfers = speed_transfers * 0;
    
    % for those who interact, add avg speed next time step
    %speed_transfers(i) = speed_xy_avg;
    speed_transfers(i) = speed_transfers(i)+(speeds(i)<speeds(j)).*(speeds(j)-speeds(i));
    
    %(i<=N1)
    
    %n_speed_transfers_total = (speeds(i)<speeds(j));
    %n_received_transfers_blue = n_speed_transfers_total(1:N1);
    


    c1 = cov(agents(idx_blue,1:2));
    c2 = cov(agents(idx_green,1:2));

    OUT_cov(:, t) = [trace(c1); trace(c2)];
    
    figure(1)
    % plot blue group
    plot(agents(idx_blue,1), agents(idx_blue,2), '.')
    hold on
    % plot green group 
    plot(agents(idx_green,1), agents(idx_green,2), '.g')
    % plot those that interact
    plot(agents(i,1), agents(i,2), 'r.')
    hold off
    
    axis([-window_size window_size -window_size window_size])
    %pause(.1)
    
%     figure(3)
%     imshow(B)

    drawnow
end

fprintf('blue group:\n')
m1 = mean(OUT_speeds(idx_blue,:),1);
m1 = median(OUT_speeds(idx_blue,:),1);
%m1 = max(OUT_speeds(idx_blue,:),1);
m_all_1 = mean(m1);
s1 = std(OUT_speeds(idx_blue,:),1);
s_all_1 = mean(s1);
c1 = cov(agents(idx_blue,1:2))
fprintf('speeds: mu=%f, std=%f\n', m_all_1, s_all_1)

fprintf('green group:\n')
m2 = mean(OUT_speeds(idx_green,:),1);
m2 = median(OUT_speeds(idx_green,:),1);
%m2 = max(OUT_speeds(idx_green,:),1);
m_all_2 = mean(m2);
s2 = std(OUT_speeds(idx_green,:),1);
s_all_2 = mean(s2);
c2 = cov(agents(idx_green,1:2))
fprintf('speeds: mu=%f, std=%f\n', m_all_2, s_all_2)






plot(repmat(1:sim_duration,N1,1), OUT_speeds(idx_blue,:), '.b')
hold on
plot(repmat(1:sim_duration,N2,1), OUT_speeds(idx_green,:), '.g')
plot(m2, 'r')
% plot(m2+s2, 'r-')
% plot(m2-s2, 'r-')


plot(m1, 'y')
% plot(m1+s1, 'y-')
% plot(m1-s1, 'y-')

figure(2)
plot(conv(OUT_cov(1,:), [1 1 1 1 1 1 1 1 1 1 1 1 1]), 'b')
hold on
plot(conv(OUT_cov(2,:), [1 1 1 1 1 1 1 1 1 1 1 1 1]), 'g')
hold off

 
