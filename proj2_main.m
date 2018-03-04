% Execution of all parts.

clc;
clear;

%% Part ii

load('mnist_all.mat')

digit = train0(1,:);
digitImage = reshape(digit,28,28);
image(rot90(flipud(digitImage),-1), colormap(gray(256)), axis square tight off);