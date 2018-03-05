function [Weights1,Weights] = part_vi(eta,Input,Layers,Target,Weights1,Weights,NumHidden,NumNeurons)
%PART_VI Trains the weights
%   

% We assume 2-norm is used to compute error
% Target should be a vector of zeros with a 1 in the position of the writen
% digit+1
Error=norm(Target-Layers(1:10,NumHidden+1));

%Using matrix multiplication to avoid for loops
%Allocating a matrix of zeros for our deltas
delta=zeros(max(NumNeurons),NumHidden+1);
delta(:,NumHidden+1)=Layers(:,NumHidden+1).*(1-Layers(:,NumHidden+1))*Error;
delta_w=eta*delta(:,NumHidden+1)*Layers(:,NumHidden)';
Weights(:,NumHidden)=Weights(:,NumHidden)+delta_w;

%We must use a different method to find the deltas now
for i=-NumHidden:-2
    j=abs(i);
    delta(:,j)=delta(:,j+1)*Weights(:,:,j).*(Layers(:,j).*(ones(max(NumNeurons),1)-Layers(:,j)));
    delta_wj=eta*delta(:,j)*Layers(:,j-1)';
    Weights(:,:,j-1)=Weights(:,:,j-1)+delta_wj;
end

% Make sure that indices are correct!
delta(:,1)=delta(:,2)*Weights(:,:,1).*(Layers(:,1).*(ones(max(NumNeurons),1)-Layers(:,1)));
delta_w1=eta*delta(:,1)*Input';
Weights1=Weights1+delta_w1;

end

