function [W,H,iterdone,costhistory] = snmf(V,a,m,iter_allowed,time_allowed)
% SNMF sparse nonnegative matrix factorization
% L1-based sparsity constraint on H.
% Normalizes W column-wise.
% a - trade-off between sparsity of H and reconstruction quality; 0 for standard NMF
%       - meant to be positive, behaviour not guaranteed for negative values
% m - inner dimension

elapsed = 0;
tic;

[n,t] = size(V);
W = (rand_init(m,n))'; % unit sum columns 
H = rand_init(m,t);

if iter_allowed == Inf;
    costhistorysize = 5000+1;
else
    costhistorysize = iter_allowed+1;
end
costhistory = zeros(1,costhistorysize) -1;

costhistory(1) = klerror(W*H,V) + a * sum(H(:));
iterdone = 0;

while iterdone < iter_allowed && elapsed < time_allowed
    
    updateW();
    updateH();
    
    iterdone = iterdone +1;
    
    if length(costhistory) < iterdone +1
        % double costhistory length
        costhistory = [costhistory, zeros(size(costhistory)) -1];
    end
    costhistory(iterdone+1) = klerror(W*H,V) + a * sum(H(:));
    
    elapsed = toc;
    
end

costhistory = costhistory(1:iterdone+1); % cut off extra space

%%% nested functions:

    function updateW()
        VC = V./(W*H + 1e-9);
        VC(V==0 & W*H==0) = 1+1e-9;
        W = W.*(VC*H')./(ones(n,1)*sum(H,2)');
        [W,H] = colsum_L_one(W,H,1); % normalize columns in W
    end

    function updateH()        
        VC = V./(W*H + 1e-9);
        VC(V==0 & W*H==0) = 1+1e-9;
        H = (H.*(W'*VC))/(1+a);
    end

end
