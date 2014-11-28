function [W,H,iterdone,costhistory] = nmfsc(V,sW,sH,m,iter_allowed,time_allowed)
%NMFsc Nonnegative matrix factorization with sparseness constraints (Hoyer)
% refactored by SPH

% m - inner dimension
% sW,sH   - sparseness of W,H, in [0,1]. (give [] if no constraint)

elapsed = 0;
tic

[n,t] = size(V);

if not(isempty(sW))
    L1a = sqrt(n)-(sqrt(n)-1)*sW;
end
if not(isempty(sH))
    L1s = sqrt(t)-(sqrt(t)-1)*sH;
end
initialize_matrices();

if iter_allowed == Inf;
    costhistorysize = 5000+1;
else
    costhistorysize = iter_allowed+1;
end
costhistory = zeros(1,costhistorysize) -1;

currentobj = euclerror(W*H,V);
costhistory(1) = currentobj;
iterdone = 0;

stepsizeW = 1; 
stepsizeH = 1;

while iterdone < iter_allowed && elapsed < time_allowed
    
    updateH()
    updateW()
    
    iterdone = iterdone +1;
    currentobj = euclerror(W*H,V);
    
    if length(costhistory) < iterdone +1
        costhistory = [costhistory, zeros(size(costhistory)) -1];
    end
    costhistory(iterdone+1) = currentobj;
    
    elapsed = toc;
    
end

costhistory = costhistory(1:iterdone+1); % cut off extra space

%%% nested functions:
    function initialize_matrices() 
        
        W = rand_init(n,m);
        H = rand_init(m,t);
        H = normalize_matrix(H,'L2');
        
        if ~isempty(sW) 
            for i=1:m
                W(:,i) = projfunc(W(:,i),L1a,1,1); 
            end
        end
        if ~isempty(sH) 
            for i=1:m
                H(i,:) = (projfunc(H(i,:)',L1s,1,1))'; 
            end
        end
        
    end

    function updateH()
        if isempty(sH)
            % Update using standard NMF multiplicative update rule
            H = H.*(W'*V)./(W'*W*H + 1e-9);

            % Renormalize so rows of H have constant energy
            norms = sqrt(sum(H'.^2));
            H = H./(norms'*ones(1,t));
            W = W.*(ones(n,1)*norms);
        else
            [H,stepsizeH] = update_with_SC(H,'H',stepsizeH);
        end
    end

    function updateW()
        if isempty(sW)
            % Update using standard NMF multiplicative update rule
            W = W.*(V*H')./(W*H*H' + 1e-9);
            % previous results were produced with this here, instead of W update: 
            %   H = H.*(W'*V)./(W'*W*H + 1e-9);
        else
            [W,stepsizeW] = update_with_SC(W,'W',stepsizeW);
        end
    end

    function [Xnew,stepsizeX_new] = update_with_SC(X,name,stepsizeX)
        stepsizeX_new = stepsizeX;
        if strcmp(name,'H')
            dX = W'*(W*H-V);
        else % name is 'W'
            dX = (W*H-V)*H';
        end
        
        begobj = euclerror(W*H,V);
        % Make sure to decrease the objective:
        while 1

            Xnew = X - stepsizeX_new*dX; % step to negative gradient, then project
            if strcmp(name,'H')
                for i=1:m
                    Xnew(i,:) = (projfunc(Xnew(i,:)',L1s,1,1))';
                end
            else
                norms = sqrt(sum(Xnew.^2));
                for i=1:m 
                    Xnew(:,i) = projfunc(Xnew(:,i),L1a*norms(i),(norms(i)^2),1); 
                end
            end
            
            if strcmp(name,'H')
                newobj = euclerror(W*Xnew,V);
            else
                newobj = euclerror(Xnew*H,V);
            end
            if newobj<begobj % objective decreased, we can continue
                break
            end
            stepsizeX_new = stepsizeX_new/2; % otherwise decrease step size
            if stepsizeX_new<1e-200 % converged
                return 
            end

        end
        stepsizeX_new = stepsizeX_new*1.2; % slightly increase the step size
    end

end
