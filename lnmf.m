function [W,H,iterdone,costhistory,kl_and_sums] = ...
    lnmf(V,m,a,b,iter_allowed,time_allowed)
% Local nonnegative matrix factorization

% m - inner dimension
% a - smoothness in W, difference between base vectors
% b - sparsity in H
% cost - what cost function to use; 'eucl' (default) or 'kl'

% a and b don't affect the algorithm's behaviour; they're only used to
% calculate the objective function. Function lnmf_costs calculates
% objective function values from kl_and_sums.

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
kl_and_sums = zeros(3,costhistorysize) -1;

[c,k] = calculate_cost();
costhistory(1) = c;
kl_and_sums(:,1) = k;
iterdone = 0;

while iterdone < iter_allowed && elapsed < time_allowed
    
    update_matrices();
    iterdone = iterdone +1;
    
    if length(costhistory) < iterdone +1
        % double costhistory and kl_and_sums lengths
        costhistory = [costhistory, zeros(size(costhistory)) -1];
        kl_and_sums = [kl_and_sums, zeros(size(kl_and_sums)) -1];
    end
    [c,k] = calculate_cost();
    costhistory(iterdone+1) = c;
    kl_and_sums(:,iterdone+1) = k;
   
    elapsed = toc;
    
end

costhistory = costhistory(1:iterdone+1); % cut off extra space
kl_and_sums = kl_and_sums(:,1:iterdone+1);

%%% nested functions:

    function [c,kl_and_sums] = calculate_cost()
        kl_and_sums = [klerror(W*H,V,0), sum(sum(W'*W)), sum(diag(H*H'))];
        c = kl_and_sums(1) + a*kl_and_sums(2) - b*kl_and_sums(3);
    end

    function update_matrices()
        VC = V./(W*H);
        VC(V==0 & W*H==0) = 1; %% ADDED by SP-H to get rid of NaN problem 
        H = sqrt(H.*(W'*VC));
		
        VC = V./(W*H);
        VC(V==0 & W*H==0) = 1; %% ADDED by SP-H to get rid of NaN problem 
        W = W.*(VC*H')./(repmat(sum(W,2),[1,m])+repmat(sum(H',1),[n,1]));
        W = W./(ones(n,1)*sum(W,1));
    end

end