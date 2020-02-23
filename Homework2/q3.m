%% Q3. Scene Classification (40 Marks)
clc;
clear;

% My matric number is A0116448A.
LWD = 48;
group_num = mod(LWD, 4) + 1;
path = append('./group_', num2str(group_num));
train_path = append(path, '/train');
val_path = append(path, '/val');

