%% 1. Set initial weights
clear;
clc;
w_init = [rand; rand];
w = w_init;
lr = 0.01;
epochs = 100;

x = [0, 0.8, 1.6, 3, 4, 5];
y = [0.5, 1, 4, 5, 6, 9];
subplot(2,1,1);
plot(x, y, '*')
X = [ones(1, length(x)); x]';
d = y';

w_hist = zeros(epochs, 2);
%% 2. Calculate w

for i = 1: epochs
    for j = 1: length(X)
        e = d(j) - w' * X(j, :)';
        w = w + lr * X(j, :)' * e;
    end
    w_hist(i,:) = w';
end

%% 3. Plot
subplot(2,1,2);
x = 1:epochs;
plot(x, w_hist(:,1)', 'r');
title('Trajectory of weights');
grid on;
hold on;
plot(x, w_hist(:,2)', 'b');


dim = [.2 .5 .3 .3];
str1 = sprintf('Initial Weights: [%.2f, %.2f]', w_init(1), w_init(2));
str2 = sprintf('Final Weights: [%.2f, %.2f]', w(1), w(2));
str = [str1 newline str2];
annotation('textbox',dim,'String', str,'FitBoxToText','on');
hold off;

x = [0, 0.8, 1.6, 3, 4, 5];
y = [0.5, 1, 4, 5, 6, 9];
subplot(2,1,1);
plot(x, y, '*')
X = [ones(1, length(x)); x]';
d = y';
grid on;
hold on;
axis([-2, 6, -1, 10]);

x = -1: 0.1:6;
y = x*(w(2)) + w(1);
plot(x, y);
str = sprintf('Q4 b) Plot');
title(str);
hold off;