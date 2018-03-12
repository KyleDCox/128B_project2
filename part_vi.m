function [Weights1,Weights] = part_vi(eta,Input,Layers,Target,Weights1,Weights,NumHidden,NumNeurons)
%PART_VI Trains the weights
%   

% We assume 2-norm is used to compute error
% Target should be a vector of zeros with a 1 in the position of the writen
% digit+1
Error=Target-Layers(1:10,NumHidden+1);

%Using matrix multiplication to avoid for loops
%Allocating a matrix of zeros for our deltas
delta=zeros(max(NumNeurons),NumHidden+1);
delta(1:10,NumHidden+1)=Layers(1:10,NumHidden+1).*(1 - Layers(1:10,NumHidden+1)).*Error;
delta_w=eta*delta(:,NumHidden+1) * Layers(:,NumHidden)';
Weights(1:NumNeurons(NumHidden+1),1:NumNeurons(NumHidden),NumHidden)= ...
    Weights(1:NumNeurons(NumHidden+1),1:NumNeurons(NumHidden),NumHidden) ...
    +delta_w(1:NumNeurons(NumHidden+1),1:NumNeurons(NumHidden));

%We must use a different method to find the deltas now
for i=-NumHidden:-2
    j=abs(i);
    %size(delta(:,j+1))
    %size(Weights(:,:,j))
    %size(Layers(:,j))
    %max(NumNeurons)
%     size(delta(1:NumNeurons(j+1),j+1)')
%     size(Weights(1:NumNeurons(j+1),1:NumNeurons(j),j))
%     size(Layers(1:NumNeurons(j),j))
%     size(ones(max(NumNeurons),1)-Layers(1:NumNeurons(j),j)))
    
    delta(1:NumNeurons(j),j)=(delta(1:NumNeurons(j+1),j+1)' ...
        *Weights(1:NumNeurons(j+1),1:NumNeurons(j),j))'.*(Layers(1:NumNeurons(j),j) ...
        .*(ones(NumNeurons(j),1)-Layers(1:NumNeurons(j),j)));
    delta_wj=eta*delta(:,j)*Layers(:,j-1)';
    Weights(1:NumNeurons(j),1:NumNeurons(j-1),j-1)= ...
        Weights(1:NumNeurons(j),1:NumNeurons(j-1),j-1)+ ...
        delta_wj(1:NumNeurons(j),1:NumNeurons(j-1));
end

% Make sure that indices are correct!
delta(1:NumNeurons(1),1)=(delta(1:NumNeurons(2),2)'*...
    Weights(1:NumNeurons(2),1:NumNeurons(1),1))'.*(Layers(1:NumNeurons(1),1)...
    .*(ones(NumNeurons(1),1)-Layers(1:NumNeurons(1),1)));
delta_w1=eta*double(delta(:,1))*double(Input');
Weights1=Weights1(1:NumNeurons(1),1:784)+delta_w1(1:NumNeurons(1),1:784);

end

