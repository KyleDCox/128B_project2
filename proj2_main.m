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
eta=0.05;


%Creating target vectors
Target = eye(10) * 0.98 + 0.01;

% Create a three dimensional matrix containting the training matrices
TRAIN(1:5421,1:784, 1) = train0(1:5421, :);
TRAIN(:, :, 2) = train1(1:5421, :);
TRAIN(:, :, 3) = train2(1:5421, :);
TRAIN(:, :, 4) = train3(1:5421, :);
TRAIN(:, :, 5) = train4(1:5421, :);
TRAIN(:, :, 6) = train5(1:5421, :);
TRAIN(:, :, 7) = train6(1:5421, :);
TRAIN(:, :, 8) = train7(1:5421, :);
TRAIN(:, :, 9) = train8(1:5421, :);
TRAIN(:, :, 10) = train9(1:5421, :);

for j=1:10
    
    % train the same amount of images for every digit
    for i=1:size(train5,1)
%    for i=1:100
        for k = 1:6
            Layers=part_iv(TRAIN(i,:, k)', Weights1, Weights, NumHidden, NumNeurons);
            [Weights1, Weights]=part_vi(eta,TRAIN(i,:, k)',Layers,Target(:,k),Weights1,Weights,NumHidden,NumNeurons);
        end
    end

end

% Construct three dimensional matrix for test images and number of test
% images
TESTNO(1) = size(test0,1);
TESTNO(2) = size(test1,1);
TESTNO(3) = size(test2,1);
TESTNO(4) = size(test3,1);
TESTNO(5) = size(test4,1);
TESTNO(6) = size(test5,1);
TESTNO(7) = size(test6,1);
TESTNO(8) = size(test7,1);
TESTNO(9) = size(test8,1);
TESTNO(10) = size(test9,1);

TEST = zeros(1135, 784, 10);
TEST(1:980, :, 1) = test0;
TEST(:, :, 2) = test1;
TEST(1:1032, :, 3) = test2;
TEST(1:1010, :, 4) = test3;
TEST(1:982, :, 5) = test4;
TEST(1:892, :, 6) = test5;
TEST(1:958, :, 7) = test6;
TEST(1:1028, :, 8) = test7;
TEST(1:974, :, 9) = test8;
TEST(1:1009, :, 10) = test9;


numCorrect=0;

for i=1:6
    for j=1:TESTNO(i)
        Layers=part_iv(TEST(j,:,i)', Weights1, Weights, NumHidden, NumNeurons);
        if max(Layers(1:10,NumHidden+1))==Layers(i,NumHidden+1)
            numCorrect=numCorrect+1;
        end
    end
end


numCorrect/sum(TESTNO)

