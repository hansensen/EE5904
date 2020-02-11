%% set values
x = [0, 0.8, 1.6, 3, 4, 5];
y = [0.5, 1, 4, 5, 6, 9];
plot(x, y, '*');

X = [ones(1, length(x)); x]';
d = y';
w = (X'*  X)^-1 * X' * d

grid on;
hold on;
axis([-2, 6, -1, 10]);

x = -1: 0.1:6;
y = x*(w(2)) + w(1);
plot(x, y);
str = sprintf('Q4 a) Plot');
title(str);
hold off;