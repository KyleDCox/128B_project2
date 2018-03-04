function [Output] = part_iv(Input,Weights1,Weights,NumHidden,NumNeurons)
%PART_IV Neural Network (Matrices)

%Empty vector to be replaced at each layer
Output=zeros(max(NumNeurons),NumHidden);

%Multiplying input by initial weights matrix
Output(:,1)=Weights1*Input;

%Iterating across the number of layers
for i=1:NumHidden
    Output(:,i+1)=part_iii(Output(:,i),Weights(:,:,i));
end

end

