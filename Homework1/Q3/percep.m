function [w_b, w_hist] = percep(w_b_init, x, d, lr, iteration)
%PERCEP Summary of this function goes here
%   Detailed explanation goes here
    [n, m] = size(x);
    x = [ones(1, m); x];
    [len, ~] = size(x);
    w_hist = zeros([len, iteration]);
    w_b = w_b_init;
    for i = 1: iteration
        v = w_b * x
        y = ones(1, length(v))
        for j = 1: length(v)
            if (v(j) < 0)
                y(j) = 0;
            end
        end
        e = d - y;
        % update w
        w_b = w_b + lr * e * x';
        w_hist(:, i) = w_b;
    end
end
