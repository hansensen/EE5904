%% task 3
% clc;
clear;
%% Load data and initialization
train = load('train.mat');
eval = load('test.mat');
x_train = train.train_data;
y_train = train.train_label;
x_eavl = eval.test_data;

% Preprocessing data
mea = mean(x_train, 2);
sd = std(x_train');
train_dim = size(x_train);
sz_tr = train_dim(2);
x_train_norm = (x_train-repmat(mea,1,sz_tr))./repmat(sd',1,sz_tr);
test_dim = size(x_eavl);
sz_ts = test_dim(2);
x_test_norm = (x_eavl-repmat(mea,1,sz_ts))./repmat(sd',1,sz_ts);

threshold = 10^(-4);
C = 0.4;
sz_c = size(C);
sz_c = sz_c(2);
tr_acc = zeros(sz_c,1);
ts_acc = zeros(sz_c,1);
p = 1;

% polynomial kernel
K = (x_train_norm' * x_train_norm + 1).^1;
K_test = (x_train_norm' * x_test_norm + 1).^1;
% check mercer's condition
eigenvalues = eig(K);
% calculate alpha
alpha = get_alpha(x_train_norm, y_train, C, K);
b0 = svm(K, y_train, alpha, threshold);

% calculate accuracy
y_pred_train = (sum((alpha .* y_train).*K)+b0)';
acc_train = mean((y_pred_train > 0) == (y_train > 0));

eval_predicted = (sum((alpha .* y_train).*K_test)+b0)';

for i = 1:sz_ts
    if eval_predicted(i) > 0
        eval_predicted(i) = 1;
    else
        eval_predicted(i) = -1;
    end
end