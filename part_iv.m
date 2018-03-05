function [Layers] = part_iv(Input,Weights1,Weights,NumHidden,NumNeurons)
%PART_IV Neural Network (Matrices)

%Empty vector to be replaced at each layer
Layers=zeros(max(NumNeurons),NumHidden+1);

%Multiplying input by initial weights matrix
Layers(:,1)=Weights1*Input;

%Iterating across the number of layers
for i=1:NumHidden
    Layers(:,i+1)=part_iii(Layers(:,i),Weights(:,:,i));
end

end

