function output = myDiffusion(input_image, K, rounds)
%MYDIFFUSION Apply the paper's two-wing modular diffusion transform.
%   Arithmetic is performed in the residue ring Z_256.

validateattributes(rounds, {'numeric'}, ...
    {'scalar', 'integer', 'nonnegative', 'finite'});

[h, w, d] = size(input_image);
sequence = double(input_image(:));
total_length = numel(sequence);
left_length = floor(total_length / 2);
right_length = total_length - left_length;

left = sequence(1:left_length);
right = sequence(left_length+1:end);
[key_left, key_right] = derive_diffusion_keys(K, total_length);

for iteration = 1:rounds
    injection_right = mod(right + key_right, 256);
    left = mod(left + injection_right(1:left_length), 256);

    injection_left = mod(left + key_left, 256);
    if right_length > left_length
        extended = zeros(right_length, 1);
        extended(1:left_length) = injection_left;
        extended(end) = injection_left(end);
        injection_left = extended;
    end
    right = mod(right + injection_left, 256);

    key_left = circshift(key_left, 13);
    key_right = circshift(key_right, 17);
end

output = uint8(reshape([left; right], h, w, d));
end
