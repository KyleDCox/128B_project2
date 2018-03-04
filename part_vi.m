function [Weights1,Weights,WeightsO] = part_vi(eta,Input,Layers,Target,Weights1,Weights,WeightsO,NumHidden,NumNeurons)
%PART_VI Trains the weights
%   

% We assume 2-norm is used to compute error
Error=norm(Target-Layers(:,NumHidden+1));
%Using matrix multiplication to avoid for loops
delta=zeros(784,NumHidden+1);
delta(:,NumHidden+1)=Layers(:,NumHidden+1).*(1-Layers(:,NumHidden+1))*Error;
delta_w=eta*delta(:,NumHidden+1)*Layers(:,NumHidden+1)';
WeightsO=WeightsO+delta_w;

Dj=delta(:,NumHidden+1)*WeightsO.*(Layers(:,NumHidden).*(ones(784,1)-Layers(:,NumHidden)));
delta(:,NumHidden)=eta*Dj*Layers(:,NumHidden+1)';
Weights(:,:,NumHidden-1)=Weights(:,:,NumHidden-1)+delta(:,NumHidden);

for i=-NumHidden+1:-2
    j=abs(i);
    delta(:,j)=delta(:,j+1)*Weights(:,:,j+1).*(Layers(:,j).*(ones(784,1)-Layers(:,j)));
    Weights(:,:,i-1)=Weights(:,:,i-1)+delta(:,j);
end

% Make sure that indices are correct!
delta(:,1)=delta(:,2)*Weights(:,:,1).*(Input.*(ones(784,1)-Input));
Weights1=Weights1+delta(:,1);

end

