function out = de_permutation(img, K)
% Inverse 3D-ATP: Z, then X, then Y.
img = double(img);
[h, w, d] = size(img);
[my, mx, mz] = build_shift_fields(K, h, w, d);
[X, Y, Z] = ndgrid(1:h, 1:w, 1:d);
s = img(sub2ind([h w d], X, Y, mod(Z - 1 - mz, d) + 1));
s = s(sub2ind([h w d], mod(X - 1 - mx, h) + 1, Y, Z));
out = s(sub2ind([h w d], X, mod(Y - 1 - my, w) + 1, Z));
end
