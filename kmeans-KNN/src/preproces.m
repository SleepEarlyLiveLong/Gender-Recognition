% preproces.m: 
%   This file is to do some pre-process for the voice dataset.
% 
%   Copyright (c) 2018 CHEN Tianyang 
%   more info contact: tychen@whu.edu.cn

%% add path
clear;close;
addpath('functions/');

%% 
% 读取降维后的数据
load('data\voice_dedimen.mat');
% 数据离散化(按字段离散化)
[m,n] = size(voice_dedimen);
voive_dedimen_discreted = zeros(m,n);
for i = 1:(n-1)
    voive_dedimen_discreted(:,i) = mydiscretization(voice_dedimen(:,i),20);
end
voive_dedimen_discreted(:,n) = voice_dedimen(:,n);
% 保存数据
save('data\voive_dedimen_discreted.mat','voive_dedimen_discreted');

%% 删除路径
rmpath('functions/');