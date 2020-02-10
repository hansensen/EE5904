%% AND
clear;
clc;

% 1. Init
lr = 0.1;
x = [0, 0, 1, 1; 0 ,1 ,0 ,1];
d = [0, 0, 0, 1];
iteration = 20;
w_b_init = [rand, rand, rand];

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
[~, n] = size(d);
for i = 1: n
    if d(i) == 1
        plot(x(1,i), x(2,i), '*');
    else
        plot(x(1,i), x(2,i), 'o');
    end
end

x = xmin:0.01:xmax;
y = -1 * x * w_b(2) / w_b(3) - w_b(1) / w_b(3);
str = sprintf('AND Decision Boundary');
title(str);
plot(x, y);
dim = [.5 .5 .3 .3];
str1 = sprintf('Initial Weights: [%.2f, %.2f, %.2f]', w_b_init(1), w_b_init(2), w_b_init(3));
str2 = sprintf('Final Weights: [%.2f, %.2f, %.2f]', w_b(1), w_b(2), w_b(3));
str = [str1 newline str2];
annotation('textbox',dim,'String', str,'FitBoxToText','on');

subplot(2,1,2);
x = 0:iteration;
plot(x, [w_b_init(1), w_hist(1,:)], 'o-');
title('Trajectory of weights');
grid on;
hold on;
plot(x, [w_b_init(2), w_hist(2,:)], '*-');
hold on;
plot(x, [w_b_init(3), w_hist(3,:)], '+-');
hold off;