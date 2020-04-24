%% RL Project -  Task 1
clc;
clear;
load task1.mat;
%%
goal_reach = 0;
num_iteration = 10;
time = zeros(num_iteration,1);
% set gamma
gamma = 0.9;
%gamma = 0.5;

% 10 runs
for run = 1:num_iteration
    fprintf('Running iteration %d.\n', run);
    tic;
    [policy, Q, reward_tot, state, state_list] = Q_learning(reward, gamma);
    time(run) = toc;
    if state == 100
        goal_reach = goal_reach + 1;
        %fprintf('  Goal reached.\n');
    else
        goal_reach = goal_reach + 0;
        %fprintf('  Goal did not reach.\n');
    end
    %fprintf('  Iteration time: %d.\n', toc);
end

time_ave = mean(time);
fprintf('Average iteration time: %d.\n', time_ave);
fprintf('Goal-reach runs: %d.\n', goal_reach);
