function [euclerror] = euclerror(Vhat,V,varargin)
% calculates Euclidean squared error
% proportional parameter: if argument '1' given, divides the error 
%   by the number of rows in V

if isempty(varargin) || varargin{1}==0
    proportional=0;
else
    proportional=1;
end
    
euclerror = 0.5*sum(sum((V-Vhat).^2)); % Hoyer: Non-negative Matrix Factorization with Sparseness Constraints
if not(isempty(proportional)) && proportional
    euclerror = euclerror / size(V,1); % divide the error by the number of data points (rows) in V
end
end