function [weights1, weights] = part_v(NumHidden, NumNeurons)
%PART_V creates the weights for the first iteration and subsequent ones
%   uses rand to create matrices of the appropriate size
%   NumHidden is the number of hidden layers
%   NumNeurons is the number of neurons in each hidden layer

% make the first weight matrix
weights1 = rand(784, NumNeurons(1));

% initialize weights matrix to all zeros
weights = zeros(max(NumNeurons), max(NumNeurons), NumHidden);

% add weights for the appropriate size
% rows is number of neurons in next layer
% columns is number of neurons in current layer
% NumNeurons must have length of 1+length(NumHidden)
for i=1:NumHidden
    weights(1:NumNeurons(i+1),1:NumNeurons(i),i) = rand(NumNeurons(i+1),NumNeurons(i));
end

end
