function out = myDeDiffusion(img, K, rounds)
% Inverse DFTF.
[h, w, d] = size(img);
x = double(img(:));
n = numel(x);
L = x(1:floor(n/2));
R = x(floor(n/2)+1:end);
[kl, kr] = derive_diffusion_keys(K, n);
KL = cell(rounds, 1); KR = cell(rounds, 1);
for t = 1:rounds
    KL{t} = kl; KR{t} = kr;
    kl = circshift(kl, 13);
    kr = circshift(kr, 17);
end
for t = rounds:-1:1
    inj = mod(L + KL{t}, 256);
    if numel(R) > numel(L), inj = [inj; inj(end)]; end
    R = mod(R - inj, 256);
    inj = mod(R + KR{t}, 256);
    L = mod(L - inj(1:numel(L)), 256);
end
out = uint8(reshape([L; R], h, w, d));
end
