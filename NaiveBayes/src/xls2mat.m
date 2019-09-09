% Pre-processing: get voive_data.mat from file 'voice.xls' for
% the following process like training, verification and testing.
% 
%   Copyright (c) 2018 CHEN Tianyang
%   more info contact: tychen@whu.edu.cn

%% add path
clear;close;
addpath('functions/');

%% 
close all;clear;
v_orig = xlsread('data\voice.xls');      % 0:male;1:female
v = v_orig;
[vm,vn] = size(v);

%%
% preparation work: fill data losts with average number
avreage = zeros(1,vn-1);
for i=1:vn-1
    avreage(i)=mean(v(:,i));
    for j=1:vm
        if v(j,i)==0
            v(j,i)=avreage(i);
        end
    end
end

% data discretization
v_d = zeros(size(v));
stepnum = 20;
for i=1:vn-1
    v_d(:,i) = mydiscretization(v(:,i),stepnum);
    fprintf('The %d row/column has been discretizd.\n',i);
end
v_d(:,vn) = v(:,vn);

% save as .mat file
preaddr = 'data\';
save([preaddr,'voive_data.mat'],'v_d');

%% remove path
rmpath('functions/');

%% 