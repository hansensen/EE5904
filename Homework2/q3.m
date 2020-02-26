%% Q3. Scene Classification (40 Marks)
clc;
clear;

% My matric number is A0116448A.
LWD = 48;
group_num = mod(LWD, 4) + 1;
path = append('./group_', num2str(group_num));
train_path = append(path, '/train/');
val_path = append(path, '/val/');

[xtrain, ytrain] = load_data(train_path);

%% a) Rosenblatt?s Perceptron

[xtrain, ytrain] = load_data(train_path);
[xtest, ytest] = load_data(val_path);

net = perceptron;
net.performFcn = 'mse';
[net,tr] = train(net, xtrain, ytrain);

% train acc
y_train_pred = net(xtrain);
acc_train = sum(y_train_pred == ytrain)/length(ytrain);
% test acc
ytest_pred = net(xtest);
acc_test = sum(ytest_pred == ytest)/length(ytest);

fprintf("Q3a. Rosenblatt?s Perceptron\ntraining accuracy: %f, validation accuracy: %f\n", ...
    acc_train, acc_test);

%% Functions
function [V, L] = load_data(path)
    V = [];
    L = [];
    file_ls = dir(path);
    extension = '.jpg';
    for i = 1:length(file_ls)
        if (endsWith(file_ls(i).name, extension))
            full_path = append(path, file_ls(i).name);
            I = double(imread(full_path));
            V = cat(2,V,I(:));
            tmp = strsplit(file_ls(i).name, {'_', '.'});
            L = cat(2,L,str2num(tmp{2}));
%             disp(size(L))
%             disp(size(V))
        end
    end

end