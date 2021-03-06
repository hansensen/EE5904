%% Q3. Scene Classification (40 Marks)
clc;
clear;

% My matric number is A0116448A.
LWD = 48;
group_num = mod(LWD, 4) + 1;
path = append('./group_', num2str(group_num));
train_path = append(path, '/train/');
val_path = append(path, '/val/');

%% a) Rosenblatt's Perceptron
[xtrain, ytrain] = load_data(train_path, 1, false);
[xval, yval] = load_data(val_path, 1, false);

[net, tr, accu_train, accu_val] = train_perceptron(xtrain, ytrain, ...
    xval, yval);

fprintf("Q3a. Rosenblatt's Perceptron\ntraining accuracy: %f, validation accuracy: %f\n", ...
    accu_train, accu_val);

%% b) Resize and PCA

lst_resize_ratio = [2, 4, 8];
fprintf("Q3a. Resize and PCA\n");

fprintf("\nWithout PCA\n");

for i = 1 : length(lst_resize_ratio)
    scale = 1 / lst_resize_ratio(i);
    [xtrain, ytrain] = load_data(train_path, scale, false);
    [xval, yval] = load_data(val_path, scale, false);

    [net, tr, accu_train, accu_val] = train_perceptron(xtrain, ytrain, ...
        xval, yval);

    fprintf("Image size: %dx%d\nTraining Acc: %f, Val Acc: %f\n", ...
    256*scale, 256*scale, accu_train, accu_val);
end

%%
fprintf("\nWith PCA\n");
for i = 1 : length(lst_resize_ratio)
    scale = 1 / lst_resize_ratio(i);
    [xtrain, ytrain] = load_data(train_path, scale, true);
    [xval, yval] = load_data(val_path, scale, true);

    [net, tr, accu_train, accu_val] = train_perceptron(xtrain, ytrain, ...
    xval, yval);

    fprintf("Image size: %dx%d\nTraining Acc: %f, Val Acc: %f\n", ...
    256*scale, 256*scale, accu_train, accu_val);
end

%% c) 
fprintf("\nc\n");
[xtrain, ytrain] = load_data(train_path, 1, false);
[xval, yval] = load_data(val_path, 1, false);
accu_train_lst = [];
accu_val_lst = [];

for n = [1:10, 20, 50, 100]
    [net, tr, accu_train, accu_val] = train_batch(n, xtrain, ytrain, xval,...
    yval, 1000, 0);
    accu_train_lst = cat(1, accu_train_lst, accu_train);
    accu_val_lst = cat(1, accu_val_lst, accu_val);

    fprintf("Hidden Neurons: %d\nTraining Acc: %f, Val Acc: %f\n", ...
    n, accu_train, accu_val);
end

%%
plot(1:1:13, accu_train_lst, 'b')
hold on;
plot(1:1:13, accu_val_lst, 'r')

%% Q3d
fprintf("d)\n")
[xtrain, ytrain] = load_data(train_path,1,false);
[xval, yval] = load_data(val_path,1,false);

lst_reg = [.1, .2, .3, .4, .5, .6, .8, .9, 1];
n = 10;
fprintf("\nHidden Neurions: %d\n", n)
for reg = lst_reg
    [net, tr, accu_train, accu_val] = train_batch(n, xtrain, ytrain, xval,...
        yval, 1000, reg);
    fprintf("Reg: %f\nTraining Acc: %f, Val Acc: %f\n", ...
        net.performParam.regularization, accu_train, accu_val);
end

%% e) MLP Sequential Mode

[xtrain, ytrain] = load_data(train_path,1,false);
[xval, yval] = load_data(val_path,1,false);

fprintf("\ne) Sequential Mode\n")
epochs = 100;

for n = [1:10, 20, 50, 100]
    [net, accu_train, accu_val] = train_seq(n, xtrain, ytrain, epochs);

    y_train_pred = net(xtrain);
    y_train_pred = y_train_pred >= 0.5;
    acc_train = sum(y_train_pred == ytrain)/length(ytrain);
    y_val_pred = net(xval);
    y_val_pred = y_val_pred >= 0.5;
    acc_val = sum(y_val_pred == yval)/length(yval);
    fprintf("# of hidden neurons:%d\n",n);
    fprintf("acc_train: %f, acc_val: %f\n",acc_train, acc_val);
end

%%
plot(1:1:length(accu_train), accu_train, 'b')
hold on;
plot(1:1:length(accu_val), accu_val, 'r')

%% Functions
function [V, L] = load_data(path, resize_ratio, apply_pca)
    V = [];
    L = [];
    file_ls = dir(path);
    extension = '.jpg';
    for i = 1:length(file_ls)
        if (endsWith(file_ls(i).name, extension))
            full_path = append(path, file_ls(i).name);
            I = double(imread(full_path));
            I = imresize(I,resize_ratio);
            if (apply_pca)
                I = pca(I);
            end
            V = cat(2,V,I(:));
            tmp = strsplit(file_ls(i).name, {'_', '.'});
            L = cat(2,L,str2num(tmp{2}));
        end
    end
end

function [net, tr, accu_train, accu_val] = train_perceptron(xtrain, ytrain,...
    xval, yval)

    net = perceptron;
    net.performFcn = 'mse';
    [net,tr] = train(net, xtrain, ytrain);

    % train acc
    ytrain_pred = net(xtrain);
    accu_train = sum(ytrain_pred == ytrain)/length(ytrain);
    % test acc
    yval_pred = net(xval);
    accu_val = sum(yval_pred == yval)/length(yval);
end

function [net, tr, accu_train, accu_val] = train_batch(n, xtrain, ytrain, xval,...
    yval, epochs, reg)

    net = patternnet(n);

    net = configure(net, xtrain, ytrain);
%     net.trainFcn = 'traingdx';
    net.performParam.regularization = reg;
%     net.trainParam.epochs = epochs;
%     net.divideParam.testRatio = 0;
%     net.divideParam.valRatio = 0.3;

    [net, tr] = train(net, xtrain, ytrain);

    % train accu
    ytrain_pred = net(xtrain);
    ytrain_pred = ytrain_pred >= 0.5;
    accu_train = sum(ytrain_pred == ytrain)/length(ytrain);
    % test accu
    ytest_pred = net(xval);
    ytest_pred = ytest_pred >= 0.5;
    accu_val = sum(ytest_pred == yval)/length(yval);
end

function [net, accu_train, accu_val] = train_seq(n, images, labels, epochs)
    % 1. Change the input to cell array form for sequential training
    images_c = num2cell(images, 1);
    labels_c = num2cell(labels, 1);
    train_num = length(labels);
    
    % 2. Construct and configure the MLP
    net = patternnet(n);
    
    net.divideFcn = 'dividetrain';
    net.trainFcn = 'traingdx';

    accu_train = zeros(epochs,1);
    accu_val = zeros(epochs,1);

    for i = 1: epochs
        display(['Epoch: ', num2str(i)])
        idx = randperm(train_num);
        net = adapt(net, images_c(:,idx), labels_c(:,idx));
%         pred_train=round(net(images(:,1:train_num)));
%         accu_train(i) = mean(abs(pred_train-labels(1:train_num)));
%         pred_val=round(net(images(:,train_num+1:end)));
%         accu_val(i) = mean(abs(pred_val-labels(train_num+1:end)));
%         fprintf("[Epoch: %d]\nTraining Acc: %f, Val Acc: %f\n", ...
%         i, accu_train(i), accu_val(i));
    end
end