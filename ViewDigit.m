function [NumIm] = ViewDigit(digit)

%Following the instructions from Greenbaum and Chartier
digitimage=reshape(digit,28,28);
NumIm=imagesc(rot90(flipud(digitimage),-1));

end

