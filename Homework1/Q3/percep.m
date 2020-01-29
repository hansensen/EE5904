function [w_b, w_hist] = percep(w_b, x, d, lr, iteration)
%PERCEP Summary of this function goes here
%   Detailed explanation goes here

    x = [ones(1, 4); x];
    size(x)
    [len, ~] = size(x);
    w_hist = zeros([len, iteration]);
    size(w_hist)
    for i = 1: iteration
        v = w_b * x;
        y = hardlim(v);
        e = d - y;
        % update w
        w_b = w_b + lr * (d -y ) * x';
        w_hist(:, i) = w_b;
    end
end

