% gender_recog.m: 
%   This file is to do the voice gender recognition through Naive Bayes.
%   Attention: both spliting training and validation sets, do the training 
%              and validation process are completed here.
%   Copyright (c) 2018 CHEN Tianyang 
%   more info contact: tychen@whu.edu.cn

%% add path
clear;close;
addpath('functions/');

%% read data (choose 1 out of 4)
% ---------------- 1- ȡ20άԭʼ(����)���� ----------------
% load('data\voive_data_init.mat');
% v_d = v_d_init;
% ---------------- 2- ȡ20ά��ɢ��������� ----------------
load('data\voive_data.mat');
% ---------------- 3- ȡ��ά���(10ά)�������� ------------
% load('data\voice_dedimen.mat');
% v_d = voice_dedimen;
% ---------------- 4- ȡ��ά���(10ά)��ɢ���� ------------
% load('data\voive_dedimen_discreted.mat');
% v_d = voive_dedimen_discreted;
% ------------------------ over --------------------------

%% �������ѵ��������֤��
[m,n] = size(v_d);
train_num = 1100;
test_num = 1584-train_num;
a=randperm(1584);
a=a(:);
b=randperm(1584)+1584;
b=b(:);
train_list_m = a(1:train_num);          % ѵ�������
train_list_f = b(1:train_num);
Train_m=v_d(train_list_m,1:(n-1));         % ѵ��������
Train_f=v_d(train_list_f,1:(n-1));
Test_m=v_d(1:1584,1:(n-1));                % ���Լ�����
Test_f=v_d(1585:3168,1:(n-1));
Test_m(train_list_m,:)=[];
Test_f(train_list_f-1584,:)=[];

% k-means ����
k=10;
DIM = 1;
errdlt = 0.5;
% ����Ů������
[Idx_m,C_m,~,~,Errlist_m] = mykmeans(Train_m,k,DIM,errdlt);
[Idx_f,C_f,~,~,Errlist_f] = mykmeans(Train_f,k,DIM,errdlt);
% figure;plot(Errlist_m,'-*');
% figure;plot(Errlist_f,'-*');
C_m = [C_m,zeros(k,1)];
C_f = [C_f,ones(k,1)];
C=[C_m;C_f];

% �ֱ����Ů���Լ���KNNʶ��
K = 9;
dists = zeros(k*2,2);
% �жϲ��Լ��е�����
P_M = 0;
N_M2M = 0;
N_M2F = 0;
for i=1:test_num
    temp = repmat(Test_m(i,:),2*k,1);
    dists(:,1) = sum((temp-C(:,1:(n-1))).^2,2);
    dists(:,2) = [zeros(k,1);ones(k,1)];
    [B,ind] = sort(dists(:,1));
    ind = ind(1:K,1);
    for j=1:K
        if ind(j,1)<=k
            P_M = P_M+1;
        end
    end
    if P_M>=(K+1)/2         % K��Ҫ������
        N_M2M = N_M2M+1;
    else
        N_M2F = N_M2F+1;
    end
    P_M = 0;
end
correct_m2m = N_M2M/test_num;

% �жϲ��Լ��е�Ů��
P_F = 0;
N_F2M = 0;
N_F2F = 0;
for i=1:test_num
    temp = repmat(Test_f(i,:),2*k,1);
    dists(:,1) = sum((temp-C(:,1:(n-1))).^2,2);
    dists(:,2) = [zeros(k,1);ones(k,1)];
    [B,ind] = sort(dists(:,1));
    ind = ind(1:K,1);
    for j=1:K
        if ind(j,1)>k && ind(j,1)<=2*k
            P_F = P_F+1;
        end
    end
    if P_F>=(K+1)/2
        N_F2F = N_F2F+1;
    else
        N_F2M = N_F2M+1;
    end
    P_F = 0;
end
correct_f2f = N_F2F/test_num;
correct_total = (N_M2M+N_F2F)/(test_num*2);

% ��ӡ���
fprintf('������Ϊ��������ȷ��: %f.\n',correct_m2m);
fprintf('Ů����ΪŮ������ȷ��: %f.\n',correct_f2f);
fprintf('�ܷ�����ȷ��: %f.\n',correct_total);

%% ɾ��·��
rmpath('functions/');