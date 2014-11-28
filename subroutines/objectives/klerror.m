function e = klerror(Vhat,V,varargin)
% calculates extended Kullback-Leibler divergence
% proportional parameter: if argument '1' given, divides the divergence 
%   by the number of rows in V

if isempty(varargin) || varargin{1}==0
    proportional=0;
else
    proportional=1;
end

    temp = V.*log(V./Vhat);
    temp(temp ~= temp) = 0; % NaN ~= NaN
    e = sum(sum(temp - V + Vhat));
    
    if proportional
        e = e / size(V,1); % divide the error by the number of data points (rows) in V
    end
end