function [key_left, key_right] = derive_diffusion_keys(K, total_length)
%DERIVE_DIFFUSION_KEYS Map chaotic values to byte driving fields.

K = double(K);
if size(K, 2) ~= 6
    error('derive_diffusion_keys:InvalidShape', ...
        'K must have six columns.');
end
if any(~isfinite(K(:)))
    error('derive_diffusion_keys:NonFiniteInput', ...
        'Diffusion key stream contains NaN or Inf.');
end

mapped = zeros(size(K));
for channel = 1:6
    maximum = max(K(:, channel));
    minimum = min(K(:, channel));
    span = maximum - minimum;
    if span <= eps(max(abs([maximum, minimum, 1])))
        error('derive_diffusion_keys:ConstantColumn', ...
            'Chaotic key column %d is constant.', channel);
    end
    normalized = (2 * K(:, channel) - (maximum + minimum)) / span;
    fractional = mod(abs(normalized), 1);
    mapped(:, channel) = mod(floor(fractional * 1e10), 256);
end

raw = mapped(:);
if numel(raw) < total_length
    error('derive_diffusion_keys:ShortKeyStream', ...
        'Diffusion requires %d bytes; received %d.', ...
        total_length, numel(raw));
end
noise = double(raw(1:total_length));

left_length = floor(total_length / 2);
key_left = noise(1:left_length);
key_right = noise(left_length+1:end);
end
