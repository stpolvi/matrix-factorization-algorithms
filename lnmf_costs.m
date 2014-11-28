function [ cost ] = lnmf_costs( kl_and_sums, a, b )
% input: [klerror, sum(sum(W'*W)), sum(diag(P*P'))], alpha, beta
% returns: klerror(estimate,V,0) + alpha*sum(sum(W'*W)) - beta*sum(diag(P*P'))

if size(kl_and_sums,2) ~= 3
    error('input argument kl_and_sums must be a 3-column matrix')
end

kl = kl_and_sums(:,1);
w_sum = kl_and_sums(:,2);
p_sum = kl_and_sums(:,3);

cost = kl + a.*w_sum - b.*p_sum;

end

