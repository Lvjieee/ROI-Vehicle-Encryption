function [kl, kr] = derive_diffusion_keys(K, n)
% Eq. (5): keystream K2 -> byte fields K_L (floor(n/2)) and K_R.
K = double(K);
m = zeros(size(K));
for c = 1:6
    hi = max(K(:, c)); lo = min(K(:, c));
    m(:, c) = mod(floor(mod(abs((2*K(:, c) - (hi + lo)) / (hi - lo)), 1) * 1e10), 256);
end
m = m(:);
kl = m(1:floor(n/2));
kr = m(floor(n/2)+1:n);
end
