function [Layers] = part_iv(Input,Weights1,Weights,NumHidden,NumNeurons)
%PART_IV Neural Network (Matrices)

%Empty vector to be replaced at each layer
Layers=zeros(max(NumNeurons),NumHidden+1);

size(Weights1);
size(Input);
NumNeurons(1);
%Multiplying input by initial weights matrix
Layers(1:NumNeurons(1),1)=part_iii(double(Input), double(Weights1));

%Iterating across the number of layers
for i=1:NumHidden
    Layers(1:NumNeurons(i+1),i+1)=part_iii(Layers(1:NumNeurons(i),i),Weights(1:NumNeurons(i+1),1:NumNeurons(i),i));
end

end

