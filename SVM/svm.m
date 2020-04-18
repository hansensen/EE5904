function b0 = svm(K, y_train, alpha, threshold)
%   Detailed explanation goes here
    % calculate alpha
    sv_idx = find(alpha > threshold);% find support vector\
    tmp =sum(alpha .*y_train .* K(:,sv_idx), 1);
    b0 = mean(y_train(sv_idx) - tmp');
end
