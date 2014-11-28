function norms = multi_norms( X, direction, norm_name)
%MULTI_NORMS Calculates the L1- or L2-norm for each row or column in X.
%   X - the matrix to be measured
%   direction - 1 for column-wise and 2 for row-wise
%   norm_name - 'L1','L2'
    
    if direction == 1
        X = X';
    end

    if strcmp(norm_name, 'L1')
        norms = sum(abs(X),2);
    elseif strcmp(norm_name, 'L2')
        norms = sqrt(sum(X.^2,2));
    else
        error(['Unsupported norm name ', norm_name])
    end
    
    if direction == 1
        norms = norms';
    end
    
end

