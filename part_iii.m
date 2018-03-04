function [OUT] = part_iii(input, weight)
%PART_III A neuron as shown in the figure
%   computes NET and then OUT as shown
%   assuems input and weight are of the same length

NET = weight*input;
OUT = (1 + exp(-NET)).^-1;

end

