function recovered = de_permutation(image, K)
%DE_PERMUTATION Invert sequential Z/X/Y fibre shifts.

image = double(image);
[h, w, d] = size(image);
[shift_y, shift_x, shift_z] = build_shift_fields(K, h, w, d);
[X, Y, Z] = ndgrid(1:h, 1:w, 1:d);

z_source = mod(Z - 1 - shift_z, d) + 1;
idx_z = sub2ind([h, w, d], X, Y, z_source);
stage_z = image(idx_z);

x_source = mod(X - 1 - shift_x, h) + 1;
idx_x = sub2ind([h, w, d], x_source, Y, Z);
stage_x = stage_z(idx_x);

y_source = mod(Y - 1 - shift_y, w) + 1;
idx_y = sub2ind([h, w, d], X, y_source, Z);
recovered = stage_x(idx_y);
end
