function [NumIm] = ViewDigit(digit)

digitimage=reshape(digit,28,28);
NumIm=image(rot90(flipud(digitimage),-1));

end

