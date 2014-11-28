function [W,H,P,iterdone,costhistory] = nmf3(V,m1,m2,normd,cost,iter_allowed,time_allowed)
% unconstrained Nonnegative matrix factorization into three factor matrices

% W*H*P approximates V.

% m1, m2 - inner dimensions
% normd - normalize 'W' (columns) or 'H' (rows); In either case, P is row-normalized
% cost - what cost function to use; 'eucl' or 'kl'

elapsed = 0;
tic;

if strcmp(normd,'W')
    norm_W = 1;
elseif strcmp(normd,'H')
    norm_W = 0;
else
    error('Specify whether to normalize W or H');
end

[n,t] = size(V);
W = rand_init(n,m1); 
H = rand_init(m1,m2); 
P = rand_init(m2,t);

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
        estimate = W*H*P;
        if strcmp(cost, 'eucl')
            c = euclerror(estimate,V);
        elseif strcmp(cost, 'kl')
            c = klerror(estimate,V);
        else
            error(['Unknown cost: ', cost]);
        end
    end

    function do_updates()
        
        updateP();
        [H,P] = rowsum_R_one(H,P); % normalize rows in P 
        
        if norm_W
            updateW();
            [W,H] = colsum_L_one(W,H); % normalize columns in W
        end
        
        updateH();
        
        if not(norm_W)
            [W,H] = rowsum_R_one(W,H); % normalize rows in H
            updateW();
        end
        
    end

    function updateW()
        HP = H*P;
        if strcmp(cost, 'eucl')
            W = W.*(V*HP')./(W*(HP*HP') + 1e-9);
        elseif strcmp(cost, 'kl')
            W = W.*((V./(W*HP + 1e-9))*HP')./(ones(n,1)*sum(HP,2)');
        end
    end

    function updateH()
        if strcmp(cost, 'eucl')
            H = H.*(W'*V*P')./(((W'*W)*H*(P*P')) + 1e-9);
        elseif strcmp(cost, 'kl')
            H = H.*(W'*(V./(W*H*P + 1e-9))*P')./((ones(m1,1)*sum(P,2)').*(sum(W,1)'*ones(1,m2)));
        end
    end

    function updateP()
        WH = W*H;
        if strcmp(cost, 'eucl')
            P = P.*(WH'*V)./((WH'*WH)*P + 1e-9);
        elseif strcmp(cost, 'kl')
            P = P.*(WH'*(V./(WH*P + 1e-9)))./(sum(WH,1)'*ones(1,t));
        end
    end

end
