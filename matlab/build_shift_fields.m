function [my, mx, mz] = build_shift_fields(K, h, w, d)
% 3D-ATP displacement fields, expanded to h x w x d.
v = mod(double(K(:)) * 1e7, 1);
kh = v(1:h); kw = v(h+1:h+w); kd = v(h+w+1:h+w+d);
my = repmat(reshape(mod(floor(kh * kd' * 1e5), w), h, 1, d), 1, w, 1);
mx = repmat(reshape(mod(floor(kw * kd' * 1e5), h), 1, w, d), h, 1, 1);
mz = repmat(mod(floor(kh * kw' * 1e5), d), 1, 1, d);
end
