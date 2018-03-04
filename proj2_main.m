% Execution of all parts.

clc;
clear;

%% Part ii

load('mnist_all.mat')

figure(1);
subplot(2,2,1);
ViewDigit(train0(1,:));
axis square tight off;

subplot(2,2,2);
ViewDigit(train1(3033,:));
axis square tight off;

subplot(2,2,3);
ViewDigit(train7(6123,:));
axis square tight off;

subplot(2,2,4);
ViewDigit(train8(949,:));
axis square tight off;
colormap(gray(256))


T=zeros(10,784);
T(1,:)=mean(train0);
T(2,:)=mean(train1);
T(3,:)=mean(train2);
T(4,:)=mean(train3);
T(5,:)=mean(train4);
T(6,:)=mean(train5);
T(7,:)=mean(train6);
T(8,:)=mean(train7);
T(9,:)=mean(train8);
T(10,:)=mean(train9);

figure(2);
subplot(2,5,1);
ViewDigit(T(1,:))
axis square tight off;

subplot(2,5,2);
ViewDigit(T(2,:))
axis square tight off;

subplot(2,5,3);
ViewDigit(T(3,:))
axis square tight off;

subplot(2,5,4);
ViewDigit(T(4,:))
axis square tight off;

subplot(2,5,5);
ViewDigit(T(5,:))
axis square tight off;

subplot(2,5,6);
ViewDigit(T(6,:))
axis square tight off;

subplot(2,5,7);
ViewDigit(T(7,:))
axis square tight off;

subplot(2,5,8);
ViewDigit(T(8,:))
axis square tight off;

subplot(2,5,9);
ViewDigit(T(9,:))
axis square tight off;

subplot(2,5,10);
ViewDigit(T(10,:))
axis square tight off;

colormap(gray(256))