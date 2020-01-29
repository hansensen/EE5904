%% OR

% 1. Init
lr = 1.0;
x = [0, 0, 1, 1; 0 ,1 ,0 ,1];
d = [0, 1, 1, 1];
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
[~, n] = size(d)
for i = 1: n
    if d(i) == 1
        plot(x(1,i), x(2,i), '*');
    else
        plot(x(1,i), x(2,i), 'o');
    end
end

x = xmin:0.01:xmax;
y = -1 * x * w_b(2) / w_b(3) - w_b(1) / w_b(3);
plot(x, y);

subplot(2,1,2);
x = 0:iteration;
plot(x, [w_b_init(1), w_hist(1,:)], 'o-');
grid on;
hold on;
plot(x, [w_b_init(2), w_hist(2,:)], '*-');
hold on;
plot(x, [w_b_init(3), w_hist(3,:)], '+-');
% axis([0, iteration+1, -10, 10]);
hold off;