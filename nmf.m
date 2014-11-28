function [W,H,iterdone,costhistory] = nmf( V,m,cost,iter_allowed,time_allowed )
% performs standard NMF 
% uses nsNMF algorithm with no nonsmoothing constraint
% normalizes H row-wise

[W,H,iterdone,costhistory] = ...
    nsnmf(V,m,0,cost,iter_allowed,time_allowed);

end

