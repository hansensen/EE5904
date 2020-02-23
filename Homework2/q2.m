%% Q2. Function Approximation (20 Marks)
clc;
clear;

% generate training and test dataset
xtrain = -1:0.05:1;
xtest = -1:0.01:1;
xout = -3:0.01:3;
ytrain = fn(xtrain);
ytest = fn(xtest);
yout = fn(xout);

lst_hidden_neurons = [1:10 20 50 100];

lr=0.00001;
epochs = 100000;

for n = lst_hidden_neurons
    net = train_seq(n, xtrain, ytrain, epochs, lr);

    ytrain_pred = net(xtrain);
    ytest_pred = net(xtest);
    yout_pred = net(xout);
    
    set(gcf, 'Position',  [100, 200, 600, 600*1.414])
    subplot(3,1,1)
    plot(xtrain, ytrain, xtrain, ytrain_pred, 'ob')
    str = sprintf('MLP: 1-%d-1 (Sequential Mode) x_{train}', n);
    title(str)
    xlim([-3 3])

    subplot(3,1,2)
    plot(xtest, ytest, xtest, ytest_pred, 'or')
    str = sprintf('MLP: 1-%d-1 (Sequential Mode) x_{test}', n);
    title(str)
    xlim([-3 3])

    subplot(3,1,3)
    plot(xout, yout, xout, yout_pred, 'or')
    str = sprintf('MLP: 1-%d-1 (Sequential Mode) x_{out}', n);
    title(str)
    xlim([-3 3])

    str = sprintf('Q2-a-%d.png', n);
    saveas(gcf, str);
end

%% Functions
function f = fn(x)
    f = 1.2*sin(pi*x) - cos(2.4*pi*x);
end

function net = train_seq(n, x, y, epochs, lr)
    trainRatio = 1;
    
    net = feedforwardnet(n, 'traingdx');
    net.layers{1}.transferFcn = 'tansig';
    net.layers{2}.transferFcn = 'purelin';
    
    net = configure(net, x, y);
    net.trainparam.lr = lr;
    net.trainparam.epochs=epochs;
    net.trainparam.goal=1e-10;
    net.divideParam.trainRatio = trainRatio;
    net.divideParam.valRatio = 1-trainRatio;
    net.divideParam.testRatio = 0;
    
    net = train(net, x, y);
end