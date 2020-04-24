%% Taks 2
clc;
clear;
load qeval.mat;
%%
tic;
gamma=0.9;
[policy, Q, reward_tot, state, state_list] = Q_learning(qevalreward, gamma);
time_spent = toc;
qevalstates = state_list';