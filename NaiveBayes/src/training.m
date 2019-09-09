% Training: train and get the model we need, then save the data as .mat.
% 
%   Copyright (c) 2018 CHEN Tianyang
%   more info contact: tychen@whu.edu.cn

%% add path
clear;close;
addpath('functions/');

%% 
% get data
load('data\voive_data.mat');
stepnum = 20;       % 连续特征离散化

% construct TrainingSet
M_train_num = 1100;
F_train_num = 1100;
train_num = M_train_num+F_train_num;
TrainingSets = repmat(struct('gender',0,'code',0,'number',0,...
    'feature',zeros(max(M_train_num,F_train_num),20),...
    'feature_prob',zeros(stepnum,20),'gender_prob',0),2,1);

%% 
% fill the TrainingSet
TrainingSets(1).gender ='male';
TrainingSets(2).gender ='female';
TrainingSets(1).code = 0;
TrainingSets(2).code = 1;
F_num = sum(v_d(:,21));
M_num = size(v_d,1)-sum(v_d(:,21));
TrainingSets(1).number = M_train_num;
TrainingSets(2).number = F_train_num;
TrainingSets(1).gender_prob = M_train_num/train_num;
TrainingSets(2).gender_prob = F_train_num/train_num;

% divide origin dataset as TrainingSet and ValidationSets(1100,484)
% TrainingSet: 1100*2
M_idx=randperm(M_num);
F_idx=randperm(F_num)+M_num;
M_train_idx=M_idx(1:M_train_num);
F_train_idx=F_idx(1:F_train_num);
TrainingSets(1).feature = v_d(M_train_idx,:);     % 1100 data for male
TrainingSets(2).feature = v_d(F_train_idx,:);     % 1100 data for female
 
% TrainingSet: data left,484*2
M_ValidationSets = v_d(1:M_num,:);
M_ValidationSets(M_train_idx,:)=[];                 % 484 data for male
F_ValidationSets = v_d(M_num+1:size(v_d,1),:);
F_ValidationSets(F_train_idx-M_num,:)=[];           % 484 data for female
ValidationSets = repmat(struct('gender',0,'code',0,'number',0,...
    'feature',zeros(max(M_num-M_train_num,F_num-F_train_num),20),...
    'results',zeros(max(M_num-M_train_num,F_num-F_train_num),3)),2,1);
ValidationSets(1).gender = 'male';
ValidationSets(2).gender = 'female';
ValidationSets(1).code = 0;
ValidationSets(2).code = 1;
ValidationSets(1).number = M_num-M_train_num;
ValidationSets(2).number = F_num-F_train_num;
ValidationSets(1).feature = M_ValidationSets;
ValidationSets(2).feature = F_ValidationSets;
ValidationSets(1).results = ones(ValidationSets(1).number,3);
ValidationSets(2).results = ones(ValidationSets(2).number,3);

%% 
% Calculate conditional probabilities
for j=1:20
    for i=1:stepnum
        TrainingSets(1).feature_prob(i,j) = ...
            (myhowmany(i,TrainingSets(1).feature(:,j))+1)/(M_train_num+1);
        TrainingSets(2).feature_prob(i,j) = ...
            (myhowmany(i,TrainingSets(2).feature(:,j))+1)/(F_train_num+1);
    end
end

%% 
% save the data package as .mat format
preaddr = 'data\';
save([preaddr,'TrainingSets.mat'],'TrainingSets');
save([preaddr,'ValidationSets.mat'],'ValidationSets');

%% remove path
rmpath('functions/');

%%