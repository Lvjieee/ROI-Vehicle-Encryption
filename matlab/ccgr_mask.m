function mask = ccgr_mask(bad, K1, rounds)
% Pixels corrupted by the 0/255 cipher bytes: spread over the DFTF wing
% pairs, then mapped back by inverse 3D-ATP.
[h, w, d] = size(bad);
m = bad(:);
n = numel(m);
L = m(1:floor(n/2));
R = m(floor(n/2)+1:end);
for t = 1:rounds
    p = L;
    if numel(R) > numel(L), p = [L; L(end)]; end
    R = R | p;
    L = L | R(1:numel(L));
end
mask = de_permutation(double(reshape([L; R], h, w, d)), K1) > 0;
end
