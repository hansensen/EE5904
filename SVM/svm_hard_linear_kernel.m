%% (i) A hard-margin SVM with the linear kernel

% clc; 
clear;
%% Load data and initialization
load train.mat
load test.mat

C = 10^6;
threshold = 10^(-4);

%% Preprocessing data
mu = mean(train_data,2);
s = std(train_data, 0, 2);
train_dim = size(train_data);
sz_tr = train_dim(2);
x_train = (train_data - repmat(mu,1, sz_tr))./repmat(s,1,sz_tr);
y_train = train_label;

test_dim = size(test_data);
sz_ts = test_dim(2);
x_test = (test_data - repmat(mu,1,sz_ts))./repmat(s,1,sz_ts);
y_test = test_label;

% linear kernel
K = x_train' * x_train; % xi*xj

%% Calculate alpha
alpha = get_alpha(x_train, y_train, C, K);

sv_idx = find(alpha > threshold);% find support vector
sv = x_train(:,sv_idx);
sv_label = y_train(sv_idx);
w0 = sum(bsxfun(@times, alpha .* y_train, x_train'),1);
b0 = mean(1./sv_label' - w0 * sv);

%% calculate accuracy
pre_train = sign(w0 * x_train + b0*ones(1,sz_tr))';
acc_train = mean(pre_train == y_train);
pre_test = sign(w0 * x_test + b0*ones(1,sz_ts))';
acc_test = mean(pre_test == y_test);

fprintf('Training acc of hardmargin linear kernel is %g.\n', acc_train);
fprintf('Testing acc of hardmargin linear kernel is %g.\n', acc_test);