function X = rand_init(n,m)
%RAND_INIT Returns a (n x m) nonnegative random matrix X.
%Rows in X are normalized.

    X = rand(n,m);
    X = normalize_matrix(X);

end

