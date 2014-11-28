function [W,H,P,iterdone,costhistory] = snmf3(V,alpha,m1,m2,iter_allowed,time_allowed)
% W*H*P approximates V, L1-based sparsity constraint on P.
% Normalizes W and H column-wise.
% alpha - trade-off between sparsity of P and reconstruction quality; 0 for standard NMF3
%       - meant to be positive, behaviour not guaranteed for negative values
% m1, m2 - inner dimensions

elapsed = 0;
tic;

[n,t] = size(V);
W = (rand_init(m1,n))'; % unit sum columns in W and H
H = (rand_init(m2,m1))';
P = rand_init(m2,t);

if iter_allowed == Inf;
    costhistorysize = 5000+1;
else
    costhistorysize = iter_allowed+1;
end
costhistory = zeros(1,costhistorysize) -1;

estimate = W*H*P;
costhistory(1) = klerror(estimate,V) + alpha * sum(P(:));
iterdone = 0;

while iterdone < iter_allowed && elapsed < time_allowed
    
    do_updates();
    iterdone = iterdone +1;
    
    if length(costhistory) < iterdone +1
        % double costhistory length
        costhistory = [costhistory, zeros(size(costhistory)) -1];
    end
    estimate = W*H*P;
    costhistory(iterdone+1) = klerror(estimate,V) + alpha * sum(P(:));
    
    elapsed = toc;
    
end

costhistory = costhistory(1:iterdone+1); % cut off extra space

%%% nested functions:

    function do_updates()
        
        updateW();
        [W,H] = colsum_L_one(W,H,1); % normalize columns in W

        updateH();
        [H,P] = colsum_L_one(H,P,1); % normalize columns in H
        
        updateP();
           
    end

    function updateW()
        HP = H*P;
        VC = V./(W*HP + 1e-9);
        VC(V==0 & W*HP==0) = 1+1e-9;
        W = W.*(VC*HP')./(ones(n,1)*sum(HP,2)');
    end

    function updateH()
        VC = V./(W*H*P + 1e-9);
        VC(V==0 & W*H*P==0) = 1+1e-9;
        H = H.*(W'*VC*P')./((ones(m1,1)*sum(P,2)').*(sum(W,1)'*ones(1,m2)));
    end

    function updateP()        
        VC = V./(W*H*P + 1e-9);
        VC(V==0 & W*H*P==0) = 1+1e-9;
        P = (P.*((W*H)'*VC))/(1+alpha);
    end

end
