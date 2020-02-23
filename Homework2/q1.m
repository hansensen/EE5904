%% Q1. Rosenbrock's Valley Problem (10 Marks)
%% Initialization

clc;
clear;

lr = 0.001;
threshold = 1e-4;
epoch_limit = 10000;
w_init = rand(1, 2) * 0.5;

%% a). Use Gradient Descent Method

w = zeros(epoch_limit,2);
w(1,:) = w_init;
f = zeros(epoch_limit,1);

for epoch=1:epoch_limit-1
    f(epoch,1) = rv_fn(w(epoch,1), w(epoch,2));
    % calculate delta_w
    delta_w = gradient_descent(w(epoch,1), w(epoch,2), lr);
    % update w
    w(epoch+1,:) = w(epoch,:)+delta_w;
end

% Plot
fprintf("The training ends at %f epoch\n", epoch);
fprintf("The solution is x:%f & y:%f with f of %f\n", ...
    [w(epoch,1),w(epoch,2),f(epoch,1)]);

clf;
subplot(2,1,1)
plot(w(:,1), w(:,2))
xlabel('X')
ylabel('Y')
title(["Gradient Descent", "Training with lr:", num2str(lr)])

subplot(2,1,2)
semilogy(f(:))
xlabel('Iterations')
ylabel('Function Value (log)')
ylim([0 100])

%% b). Use Newton's method

w = zeros(epoch_limit,2);
w(1,:) = w_init;
f = zeros(epoch_limit,1);

for epoch=1:epoch_limit-1
    f(epoch,1) = rv_fn(w(epoch,1), w(epoch,2));
    % calculate delta_w
    delta_w = newtons_method(w(epoch,1), w(epoch,2), lr);
    % update w
    w(epoch+1,:) = w(epoch,:)+delta_w;
end

% Plot
fprintf("The training ends at %f epoch\n", epoch);
fprintf("The solution is x:%f & y:%f with f of %f\n", ...
    [w(epoch,1),w(epoch,2),f(epoch,1)]);

clf;
subplot(2,1,1)
plot(w(:,1), w(:,2))
xlabel('X')
ylabel('Y')
title(["Newton's Method", "Training with lr:", num2str(lr)])

subplot(2,1,2)
semilogy(f(:))
xlabel('Iterations')
ylabel('Function Value (log)')
ylim([0 100])

%% Define functions

% Rosenbrock's Valley Function
function f = rv_fn(x,y)
    f = (1 - x)^2 + 100 * ( y - x^2)^2;
end

% Gradient
function g = rv_g(x,y)
    g = zeros(2,1);
    g(1,1) = 2*(x-1) - 400 * x * (y - x^2); % df/dx
    g(2,1) = 200*(y-x^2); % df/dy
end

% Hessian
function H = rv_h(x,y)
    H = zeros(2,2);
    H(1,1) = 2 + 1200*x^2 - 400*y;
    H(1,2) = -400*x;
    H(2,1) = -400*x;
    H(2,2) = 200;
end

function delta_w = gradient_descent(x, y, lr)
    % calculate gradient
    g = rv_g(x, y);
    delta_w = -lr*g';
end

function delta_w = newtons_method(x, y, lr)
    % calculate gradient
    g = rv_g(x,y);
    % calculate Hessian
    H = rv_h(x,y);
    delta_w = (-inv(H)*g)';
end
