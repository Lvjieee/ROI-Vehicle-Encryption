function out = myDiffusion(img, K, rounds)
% DFTF, Eqs. (6)-(7).
[h, w, d] = size(img);
x = double(img(:));
n = numel(x);
L = x(1:floor(n/2));
R = x(floor(n/2)+1:end);
[kl, kr] = derive_diffusion_keys(K, n);
for t = 1:rounds
    inj = mod(R + kr, 256);
    L = mod(L + inj(1:numel(L)), 256);
    inj = mod(L + kl, 256);
    if numel(R) > numel(L), inj = [inj; inj(end)]; end
    R = mod(R + inj, 256);
    kl = circshift(kl, 13);
    kr = circshift(kr, 17);
end
out = uint8(reshape([L; R], h, w, d));
end
