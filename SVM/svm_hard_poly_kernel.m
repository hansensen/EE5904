%% (i) A hard-margin SVM with the linear kernel

% clc; 
clear;
%% Load data and initialization
train = load('train.mat');
test = load('test.mat');
x_train = train.train_data;
y_train = train.train_label;
x_test = test.test_data;
y_test = test.test_label;


% Preprocessing data
mea = mean(x_train, 2);
sd = std(x_train');
train_dim = size(x_train);
sz_tr = train_dim(2);
x_train_norm = (x_train-repmat(mea,1,sz_tr))./repmat(sd',1,sz_tr);
test_dim = size(x_test);
sz_ts = test_dim(2);
x_test_norm = (x_test-repmat(mea,1,sz_ts))./repmat(sd',1,sz_ts);

C = 10^6;
threshold = 10^(-4);
p = [2, 3, 4, 5];

for i = 1:length(p)
    % polynomial kernel
    K = (x_train_norm' * x_train_norm + 1).^p(i);
    K_test = (x_train_norm' * x_test_norm + 1).^p(i);

    % calculate alpha
    alpha = get_alpha(x_train_norm, y_train, C, K);
    sv_idx = find(alpha > threshold);% find support vector
    sv = x_train_norm(:,sv_idx);
    tmp =sum(alpha .*y_train .* K(:,sv_idx), 1);
    b0 = mean(y_train(sv_idx) - tmp');

    % calculate accuracy
    y_pred_train = (sum((alpha .* y_train).*K)+b0)';
    acc_train = mean((y_pred_train > 0) == (y_train > 0));

    y_pred_test = (sum((alpha .* y_train).*K_test)+b0)';
    acc_test = mean((y_pred_test > 0) == (y_test>0));

    fprintf('Training acc of hardmargin polynomial kernel of %d is %g.\n', p(i), acc_train);
    fprintf('Testing acc of hardmargin polynomial kernel of %d is %g.\n', p(i), acc_test);
end
