function [shift_y, shift_x, shift_z] = build_shift_fields(K, h, w, d)
%BUILD_SHIFT_FIELDS Construct the three deterministic 3D-ATP shift fields.

values = double(K(:));
needed = h + w + d;
if numel(values) < needed
    error('build_shift_fields:ShortKeyStream', ...
        'Permutation requires at least %d values; received %d.', ...
        needed, numel(values));
end
if any(~isfinite(values))
    error('build_shift_fields:NonFiniteKeyStream', ...
        'Permutation key stream contains NaN or Inf.');
end

fractional = mod(values * 1e7, 1);
k_h = fractional(1:h);
k_w = fractional(h+1:h+w);
k_d = fractional(h+w+1:h+w+d);

shift_y_2d = mod(floor((k_h * k_d') * 1e5), w);
shift_x_2d = mod(floor((k_w * k_d') * 1e5), h);
shift_z_2d = mod(floor((k_h * k_w') * 1e5), d);

shift_y = repmat(reshape(shift_y_2d, h, 1, d), 1, w, 1);
shift_x = repmat(reshape(shift_x_2d, 1, w, d), h, 1, 1);
shift_z = repmat(shift_z_2d, 1, 1, d);
end
