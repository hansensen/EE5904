%% a soft-margin SVM with the polynomial kernel
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

threshold = 10^(-4);
C = 0.1:0.05:3;
sz_c = size(C);
sz_c = sz_c(2);
tr_acc = zeros(sz_c,1);
ts_acc = zeros(sz_c,1);
p = 1;
for j = 1:length(C)
    % polynomial kernel
    K = (x_train_norm' * x_train_norm + 1).^1;
    K_test = (x_train_norm' * x_test_norm + 1).^1;
    % check mercer's condition
    eigenvalues = eig(K);
    % calculate alpha
    alpha = get_alpha(x_train_norm, y_train, C(j), K);
    b0 = svm(K, y_train, alpha, threshold);

    % calculate accuracy
    y_pred_train = (sum((alpha .* y_train).*K)+b0)';
    acc_train = mean((y_pred_train > 0) == (y_train > 0));

    y_pred_test = (sum((alpha .* y_train).*K_test)+b0)';
    acc_test = mean((y_pred_test > 0) == (y_test>0));

    tr_acc(j) = acc_train;
    ts_acc(j) = acc_test;

    fprintf('Training acc of softmargin polynomial kernel of C=%d is %g.\n', C(j), acc_train);
    fprintf('Testing acc of softmargin polynomial kernel of C=%d is %g.\n', C(j), acc_test);
end

plot(C, ts_acc);
xlabel('C')
ylabel('Test Accuracy')