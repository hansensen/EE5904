%% 1. Set initial weights
clear;
clc;
w_init = [rand; rand];

lr_arr = [0.0001, 0.001, 0.01, 0.1];
epochs = 1000;

w_lr = zeros(length(lr_arr), 2);

for k = 1: length(lr_arr)
    lr = lr_arr(k);
    x = [0, 0.8, 1.6, 3, 4, 5];
    y = [0.5, 1, 4, 5, 6, 9];
    X = [ones(1, length(x)); x]';
    d = y';

    w_hist = zeros(epochs, 2);
    %% 2. Calculate w
    w = w_init;
    for i = 1: epochs
        w_hist(i,:) = w';
        for j = 1: length(X)
            e = d(j) - w' * X(j, :)';
            w = w + lr * X(j, :)' * e;
        end
    end

    %% 3. Plot
    subplot(length(lr_arr),1,k);
    x = 1:epochs;
    axis([0 100 0 3])
    plot(x, w_hist(:,1)', 'r');
    title('Trajectory of weights');
    grid on;
    hold on;
    plot(x, w_hist(:,2)', 'b');
    w_lr(k,:) = w';
end
hold off;

%%
figure;
x = [0, 0.8, 1.6, 3, 4, 5];
y = [0.5, 1, 4, 5, 6, 9];
plot(x, y, '*');
X = [ones(1, length(x)); x]';
d = y';
grid on;
hold on;
axis([-2, 6, -1, 10]);

for k = 1: length(lr_arr)
    w = w_lr(k,:)';
    x = -1: 0.1:6;
    y = x*(w(2)) + w(1);
    legend()
    plot(x, y);
    hold on;
end

lgd = legend;
lgd.FontSize = 14;
legend('Data Points','lr=0.0001', 'lr = 0.001', 'lr = 0.01', 'lr = 0.1');

str = sprintf('Q4 b) Plot');
title(str);
hold off;