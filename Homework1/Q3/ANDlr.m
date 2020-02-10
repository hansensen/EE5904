%% AND
clear;
clc;

% 1. Init
lr = [0.001, 0.01, 0.1, 1, 1.5, 2, 5, 10];
x = [0, 0, 1, 1; 0 ,1 ,0 ,1];
d = [0, 0, 0, 1];
iteration = 20;
w_b_init = [rand, rand, rand];



% 3. Plot the graph
figure;
% 2. Get w and b

for k =  1: length(lr)
    [w_b, w_hist] = percep(w_b_init, x, d, lr(k), iteration);
    subplot(length(lr),1,k);
    x_label = 0:iteration;
    plot(x_label, [w_b_init(1), w_hist(1,:)], 'o-');
    str = sprintf('Trajectory of weights, lr=%.3f', lr(k));
    title(str);
    grid on;
    hold on;
    plot(x_label, [w_b_init(2), w_hist(2,:)], '*-');
    hold on;
    plot(x_label, [w_b_init(3), w_hist(3,:)], '+-');
    hold off;
end


