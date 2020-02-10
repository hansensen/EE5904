%% COMPLEMENT
clear;
clc;

% 1. Init
lr = 1.0;
x = [0, 1];
d = [1, 0];
iteration = 20;
w_b_init = [rand, rand];

% 2. Get w and b
[w_b, w_hist] = percep(w_b_init, x, d, lr, iteration);

% 3. Plot the graph
figure;
xmin = -0.5;
xmax = 3;
ymin = -0.5;
ymax = 3;
subplot(2,1,1);

axis([xmin, xmax, ymin, ymax]);
grid on;
hold on;
[~, n] = size(d)
for i = 1: n
    if d(i) == 1
        plot(x(i), 0, '*');
    else
        plot(x(i), 0, 'o');
    end
end

xline(w_b(1),'-',{'Decision Boundary'});
str = sprintf('COMPLEMENT: x=%.2f', w_b(1));
title(str);
dim = [.5 .5 .3 .3];
str1 = sprintf('Initial Weights: [%.2f, %.2f]', w_b_init(1), w_b_init(2));
str2 = sprintf('Final Weights: [%.2f, %.2f]', w_b(1), w_b(2));
str = [str1 newline str2];
annotation('textbox',dim,'String', str,'FitBoxToText','on');

subplot(2,1,2);
x = 0:iteration;
title('Trajectory of weights');
plot(x, [w_b_init(1), w_hist(1,:)], 'o-');
grid on;
hold on;
plot(x, [w_b_init(2), w_hist(2,:)], '*-');
hold off;