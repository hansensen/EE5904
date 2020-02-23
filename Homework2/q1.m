%% Q1. Rosenbrock's Valley Problem (10 Marks)
%% Initialization

clc;
clear;

lr = 0.2;
epoch_limit = 100000;
w_init = rand(1, 2) * 0.5;
threshold = 1e-20;

%% a). Use Gradient Descent Method

w = zeros(epoch_limit,2);
w(1,:) = w_init;
f = zeros(epoch_limit,1);

for epoch=1:epoch_limit
    f(epoch) = rv_fn(w(epoch,1), w(epoch,2));
    % calculate delta_w
    delta_w = gradient_descent(w(epoch,1), w(epoch,2), lr);
    % update w
    w(epoch+1,:) = w(epoch,:)+delta_w;
    if epoch > 1
        if (abs(f(epoch) - f(epoch-1)) < threshold)
            break;
        end
    end
end

set(gcf, 'Position',  [100, 200, 600, 600*1.414])
% Plot the graph
str1 = sprintf("Gradient Descent (lr=%f):", lr);
str2 = sprintf("Threshold: %.0d", threshold);
str3 = sprintf("Epochs: [%d/%d]", epoch, epoch_limit);
str4 = sprintf("x=%f, y=%f, f(x,y)=:%.0d", w(epoch,1),w(epoch,2),f(epoch,1));
str = [str1 str2 str3 str4];

subplot(2,1,1)
plot(w(1:epoch,1), w(1:epoch,2))
xlabel('x');
ylabel('y');
title(["a). Gradient Descent", "Trajectory Graph"])

subplot(2,1,2)
semilogy(f(:))
xlabel('Iterations')
ylabel('Function Value (log scale)')


dim = [0.2 0.6 0.3 0.3];
annotation('textbox',dim,'String', str,'FitBoxToText','on');


%% b). Use Newton's method
%% Initialization

clc;
clear;

lr = 0.001;
epoch_limit = 1000;
w_init = rand(1, 2) * 0.5;
threshold = 1e-20;

w = zeros(epoch_limit,2);
w(1,:) = w_init;
f = zeros(epoch_limit,1);

for epoch=1:epoch_limit-1
    f(epoch,1) = rv_fn(w(epoch,1), w(epoch,2));
    % calculate delta_w
    delta_w = newtons_method(w(epoch,1), w(epoch,2), lr);
    % update w
    w(epoch+1,:) = w(epoch,:)+delta_w;
    if epoch > 1
        if (abs(f(epoch) - f(epoch-1)) < threshold)
            break;
        end
    end
end


set(gcf, 'Position',  [100, 200, 600, 600*1.414])
% Plot the graph
str1 = sprintf("Newton's method (lr=%f):", lr);
str2 = sprintf("Threshold: %.0d", threshold);
str3 = sprintf("Epochs: [%d/%d]", epoch, epoch_limit);
str4 = sprintf("x=%f, y=%f, f(x,y)=:%.0d", w(epoch,1),w(epoch,2),f(epoch,1));
str = [str1 str2 str3 str4];

subplot(2,1,1)
plot(w(1:epoch,1), w(1:epoch,2))
xlabel('x');
ylabel('y');
title(["b). Newton's method", "Trajectory Graph"])

subplot(2,1,2)
semilogy(f(:))
xlabel('Iterations')
ylabel('Function Value (log scale)')


dim = [0.2 0.6 0.3 0.3];
annotation('textbox',dim,'String', str,'FitBoxToText','on');

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
