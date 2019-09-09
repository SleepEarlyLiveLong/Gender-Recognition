% preproces.m: 
%   This file is to do some pre-process for the voice dataset.
% 
%   Copyright (c) 2018 CHEN Tianyang 
%   more info contact: tychen@whu.edu.cn

%% add path
clear;close;
addpath('functions/');

%% 
% ��ȡ��ά�������
load('data\voice_dedimen.mat');
% ������ɢ��(���ֶ���ɢ��)
[m,n] = size(voice_dedimen);
voive_dedimen_discreted = zeros(m,n);
for i = 1:(n-1)
    voive_dedimen_discreted(:,i) = mydiscretization(voice_dedimen(:,i),20);
end
voive_dedimen_discreted(:,n) = voice_dedimen(:,n);
% ��������
save('data\voive_dedimen_discreted.mat','voive_dedimen_discreted');

%% ɾ��·��
rmpath('functions/');