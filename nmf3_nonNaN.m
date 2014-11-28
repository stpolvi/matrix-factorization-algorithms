function [W,H,P,iterdone,costhistory] = nmf3_nonNaN(V,m1,m2,normd,cost,iter_allowed,time_allowed)
% unconstrained Nonnegative matrix factorization into three factor matrices
% modified version that stops before any NaN values appear in the result matrices
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
able_to_update = 1;

while able_to_update && iterdone < iter_allowed && elapsed < time_allowed
    
    able_to_update = do_updates();
    
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

    function able_to_update = do_updates()
        
        P2 = updateP();
        if any(isnan(P2(:)))
            able_to_update = 0;
            return
        end
        P = P2;
        
        [H,P] = rowsum_R_one(H,P); % normalize rows in P 
        
        if norm_W
            W2 = updateW();
            if any(isnan(W2(:)))
                able_to_update = 0;
                return
            end
            W = W2;
            [W,H] = colsum_L_one(W,H); % normalize columns in W
        end
        
        H2 = updateH();
        if any(isnan(H2(:)))
            able_to_update = 0;
            return
        end
        H = H2;
        
        if not(norm_W)
            [W,H] = rowsum_R_one(W,H); % normalize rows in H
            W2 = updateW();
            if any(isnan(W2(:)))
                able_to_update = 0;
                return
            end
            W = W2;
        end
        
        able_to_update = 1;
        
    end

    function W2 = updateW()
        HP = H*P;
        if strcmp(cost, 'eucl')
            W2 = W.*(V*HP')./(W*(HP*HP') + 1e-9);
        elseif strcmp(cost, 'kl')
            W2 = W.*((V./(W*HP + 1e-9))*HP')./(ones(n,1)*sum(HP,2)');
        end
    end

    function H2 = updateH()
        if strcmp(cost, 'eucl')
            H2 = H.*(W'*V*P')./(((W'*W)*H*(P*P')) + 1e-9);
        elseif strcmp(cost, 'kl')
            H2 = H.*(W'*(V./(W*H*P + 1e-9))*P')./((ones(m1,1)*sum(P,2)').*(sum(W,1)'*ones(1,m2)));
        end
    end

    function P2 = updateP()
        WH = W*H;
        if strcmp(cost, 'eucl')
            P2 = P.*(WH'*V)./((WH'*WH)*P + 1e-9);
        elseif strcmp(cost, 'kl')
            P2 = P.*(WH'*(V./(WH*P + 1e-9)))./(sum(WH,1)'*ones(1,t));
        end
    end

end
