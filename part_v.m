function [weights1, weights] = part_v(NumHidden, NumNeurons)
%PART_V creates the weights for the first iteration and subsequent ones
%   uses rand to create matrices of the appropriate size
%   NumHidden is the number of hidden layers
%   NumNeurons is the number of neurons in each hidden layer

% make the first weight matrix
weights1 = rand(784, max(NumNeurons));

% initialize weights matrix to all zeros
weights = zeros(max(NumNeurons), max(NumNeurons), NumHidden);

% add weights for the appropriate size
% rows is number of neurons in next layer
% columns is number of neurons in current layer
for i=1:NumHidden - 1
    weights(NumNeurons(i+1),NumNeurons(i),i) = rand(NumNeurons(i+1),NumNeurons(i));
end
% we know that output layer has 784 neurons
weights(784,NumNeurons(size(NumNeurons)), NumHidden) = rand(784, NumNeurons(size(NumNeurons)));
