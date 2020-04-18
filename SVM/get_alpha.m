function alpha = get_alpha(train_data_std, train_label, C, K)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    train_dim = size(train_data_std);
    sz_tr = train_dim(2);
    H = train_label*train_label'.*K;

    %% Solve the problem
    f = -1 * ones(sz_tr, 1);
    Aeq = train_label';
    beq = 0;
    ub = C * ones(sz_tr, 1);
    lb=zeros(sz_tr,1);
    x0=[];
    A=[];
    b=[];
%     options = optimset('LargeScale','off','MaxIter',1000, 'Algorithm','interior-point-convex');
    options = optimset('LargeScale','off','MaxIter',1000);
    alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options);
end

% 
%     H = y_train * y_train'.*K;
%     Aeq = y_train';
%     beq = 0;
%     C = 1000000;
%     lb = zeros(2000,1);
%     ub = ones(2000,1)*C;
%     x0 = [];
%     f = -ones(2000,1);
%     A = [];
%     b = [];
%     options = optimset('LargeScale','off','MaxIter',1000);
%     alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options);