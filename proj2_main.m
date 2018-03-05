% Execution of all parts.

clc;
clear;

%% Part ii

%Loading the dataset
load('mnist_all.mat')
%Normalizing all of the inputs to greyscale
train0=train0/256;
train1=train1/256;
train2=train2/256;
train3=train3/256;
train4=train4/256;
train5=train5/256;
train6=train6/256;
train7=train7/256;
train8=train8/256;
train9=train9/256;

%Plotting example numbers fromt the dataset to test the function
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
%Changing the colors to greyscale
colormap(gray(256))

% Creating a matrix of number averages
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

%Making a subplot of the number averages
figure(2);
subplot(2,5,1);
ViewDigit(T(1,:));
axis square tight off;

subplot(2,5,2);
ViewDigit(T(2,:));
axis square tight off;

subplot(2,5,3);
ViewDigit(T(3,:));
axis square tight off;

subplot(2,5,4);
ViewDigit(T(4,:));
axis square tight off;

subplot(2,5,5);
ViewDigit(T(5,:));
axis square tight off;

subplot(2,5,6);
ViewDigit(T(6,:));
axis square tight off;

subplot(2,5,7);
ViewDigit(T(7,:));
axis square tight off;

subplot(2,5,8);
ViewDigit(T(8,:));
axis square tight off;

subplot(2,5,9);
ViewDigit(T(9,:));
axis square tight off;

subplot(2,5,10);
ViewDigit(T(10,:));
axis square tight off;

%Converting the color to greyscale
colormap(gray(256))

%% Part iv

%The Neural Net Function is Given in part_iv.m



%% Part v


%% Part vi

%The reverse pass is given in part_vi.m

%% Part vii
clc

%The Number of Hidden Layers
NumHidden = 2;

%The Number of Neurons in Each Layer
NumNeurons = [100 50 10];

%Creating random weights
[Weights1, Weights] = part_v(NumHidden, NumNeurons);

%Training Rate
eta=0.01;

% Training for 0
%Creating target vector
Target=zeros(10,1);
Target(1)=1;

numTestacc=0;

for j=1:5

    avg_out = zeros(1);
    for i=1:size(train0,1)
        Layers=part_iv(train0(i,:)', Weights1, Weights, NumHidden, NumNeurons);
        avg_out = avg_out + Layers;
    end
    Layers = avg_out / size(train0, 1);
    [Weights1, Weights]=part_vi(eta,T(1,:)',Layers,Target,Weights1,Weights,NumHidden,NumNeurons);
end
    
numCorrect=0;

for i=1:size(test0,1)
    Layers=part_iv(test0(i,:)', Weights1, Weights, NumHidden, NumNeurons);
    if max(Layers(1:10,NumHidden+1))==Layers(1,NumHidden+1)
        numCorrect=numCorrect+1;
    end
end
numCorrect/size(test0,1)

