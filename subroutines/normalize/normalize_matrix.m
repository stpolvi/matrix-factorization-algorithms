function normalized = normalize_matrix(X, varargin)
% input args: X - the matrix to be normalized
%    additionally, -number 2 for row-wise (default), 1 for column-wise normalization.
%                  -in respect to which measure of length the matrix is to be normalized:
%                   'L1' (default), 'L2', 'squared'

    if nargin > 3
        error(['Too many input arguments (',num2str(nargin),')'])
    end
    
    direction = 2;
    measure = 'L1';
    
    if nargin == 3
        direction = varargin{1};
        measure = varargin{2};
    elseif not(isempty(varargin))
        if strcmp(varargin{1},'L2') || strcmp(varargin{1},'L1')
            measure = varargin{1};
        elseif varargin{1}==1 || varargin{1}==2
            direction = varargin{1};
        else
            error(['wrong argument ',varargin{1}])
        end
    end
    
    norms = multi_norms(X,direction,measure);
    coeff = 1./norms;
    
    if direction == 1
        normalized = X * diag(coeff);
        %normalized(:,all(X,1)==0) = 0; % keep 0-columns as they are
    else
        normalized = diag(coeff) * X;
        %normalized(all(X,2)==0,:) = 0; % keep 0-rows as they are
    end

end

