function [Output] = part_iv(Input,Weights1,Weights,NumHidden,NumNeurons)
%PART_IV Neural Network (Matrices)

%Empty vector to be replaced at each layer
CurrentLayer=zeros(max(NumNeurons),1);

%Multiplying input by initial weights matrix
CurrentLayer=Weights1*Input;

%Iterating across the number of layers
for i=1:NumHidden
    CurrentLayer=Weights(i)*CurrentLayer;
end

%Outputting the guess
Output=max(CurrentLayer(1:10));


end
