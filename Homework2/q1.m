%% Q1. Rosenbrock's Valley Problem (10 Marks)

clc;
clear;
%% a). Use Gradient Descent Method

lr = 0.001;
threshold = 1e-4;
epoch_limit = 10000;
w_init = rand(1, 2) * 0.5;
w = zeros(epoch_limit,2);
w(1,:) = w_init;
f = zeros(epoch_limit,1);
epoch = 1;

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
