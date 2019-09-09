% Validation: validate model achieved by testing sets with validation sets.
% 
%   Copyright (c) 2018 CHEN Tianyang
%   more info contact: tychen@whu.edu.cn

%% add path
clear;close;
addpath('functions/');

%% 
% get data
load('data/ValidationSets.mat');
load('data/TrainingSets.mat');
stepnum = 20;

%% 
% validate
prob_m = 1;
prob_f = 1;
for i=1:2
    for j=1:ValidationSets(i).number              % for each voice
        data = ValidationSets(i).feature(j,:);
        for k=1:20
            % probability of being male voice
            ValidationSets(i).results(j,1)=...
                TrainingSets(1).feature_prob(data(k),k)*ValidationSets(i).results(j,1);
            % probability of being female voice
            ValidationSets(i).results(j,2)=...
                TrainingSets(2).feature_prob(data(k),k)*ValidationSets(i).results(j,2);
        end
        if ValidationSets(i).results(j,1) > ValidationSets(i).results(j,2)
            % this is male voice
            ValidationSets(i).results(j,3) = 0;
        else
            % this is female voice
            ValidationSets(i).results(j,3) = 1;
        end
    end
end

% accuracy for male and female in confusion matrix
label_real = int8([zeros(ValidationSets(1).number,1);ones(ValidationSets(1).number,1)]);
label_predict = int8([ValidationSets(1).results(:,3);ValidationSets(2).results(:,3)]);
mtx_cfs = mycfsmtx(label_real+1,label_predict+1);

%% remove path
rmpath('functions/');

%% 