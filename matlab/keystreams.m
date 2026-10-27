function [K1, K2] = keystreams(digest, key_c, h, w, d)
% Trajectory after 5000 transient steps, split into K1 (3D-ATP) and K2 (DFTF).
r1 = ceil(h*w*d / 6);
r2 = ceil(h*w / 2);
y = rossler(generate_init(digest, key_c), r1 + r2 + 5000);
y = y(5001:end, :);
K1 = y(1:r1, :);
K2 = y(r1+1:end, :);
end
