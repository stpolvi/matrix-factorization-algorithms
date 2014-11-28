function s = sparsity(A)
n = size(A,2);
r = size(A,1);
s = 0;
for i=1:r
    L1 = sum(abs(A(i,:)));
    L2 = sqrt(sum(A(i,:).*A(i,:)));
    s = s + ((sqrt(n)-(L1/L2)) / (sqrt(n)-1)); %Hoyer 2004
end
s = s / r;
end