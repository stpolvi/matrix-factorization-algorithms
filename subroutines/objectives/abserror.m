function [abserror] = abserror(Vhat,V)
% calculates absolute error
    
abserror = sum(abs(Vhat(:)-V(:)));

end