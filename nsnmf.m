function [W,H,iterdone,costhistory] = nsnmf(V,m,ns,cost,iter_allowed,time_allowed)
% nonsmooth Nonnegative matrix factorization, nsNMF by Pascual-Montano et al.

% m - inner dimension
% ns - decides the degree in [0,1] of nonsmoothing (use 0 for standard NMF)
% cost - what cost function to use; 'eucl' (default) or 'kl'

I = eye(m);
S = (1-ns)*I + (ns/m)*ones(m);

elapsed = 0;
tic;

[n,t] = size(V);
W = rand_init(n,m);
H = rand_init(m,t);

if iter_allowed == Inf;
    costhistorysize = 5000+1;
else
    costhistorysize = iter_allowed+1;
end
costhistory = zeros(1,costhistorysize) -1;

costhistory(1) = calculate_cost();
iterdone = 0;

while iterdone < iter_allowed && elapsed < time_allowed
    
    do_updates();
    iterdone = iterdone +1;
    
    if length(costhistory) < iterdone +1
        % double costhistory length
        costhistory = [costhistory, zeros(size(costhistory)) -1];
    end
    costhistory(iterdone+1) = calculate_cost();
   
    elapsed = toc;
    
end

costhistory = costhistory(1:iterdone+1); % cut off extra space

%%% nested functions:

    function c = calculate_cost()
        estimate = W*S*H;
        if strcmp(cost, 'eucl')
            c = euclerror(estimate,V);
        elseif strcmp(cost, 'kl')
            c = klerror(estimate,V);
        else
            error(['Unknown cost: ', cost]);
        end
    end

    function do_updates()
        updateH();
        [W,H] = rowsum_R_one(W,H); % normalize rows in H
        updateW();
    end

    function updateW()
        SH = S*H;
        if strcmp(cost, 'eucl')
            W = W.*(V*SH')./(W*(SH*SH') + 1e-9);
        elseif strcmp(cost, 'kl')
            W = W.*((V./(W*SH + 1e-9))*SH')./(ones(n,1)*sum(SH,2)');
        end
    end

    function updateH()
        WS = W*S;
        if strcmp(cost, 'eucl')
            H = H.*(WS'*V)./((WS'*WS)*H + 1e-9);
        elseif strcmp(cost, 'kl')
            H = H.*(WS'*(V./(WS*H + 1e-9)))./(sum(WS,1)'*ones(1,t));
        end
    end

end