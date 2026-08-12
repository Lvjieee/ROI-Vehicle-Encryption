function output = myDeDiffusion(cipher_image, K, rounds)
%MYDEDIFFUSION Exactly invert MYDIFFUSION.

validateattributes(rounds, {'numeric'}, ...
    {'scalar', 'integer', 'nonnegative', 'finite'});

[h, w, d] = size(cipher_image);
sequence = double(cipher_image(:));
total_length = numel(sequence);
left_length = floor(total_length / 2);
right_length = total_length - left_length;

left = sequence(1:left_length);
right = sequence(left_length+1:end);
[initial_left, initial_right] = derive_diffusion_keys(K, total_length);

left_history = cell(rounds, 1);
right_history = cell(rounds, 1);
key_left = initial_left;
key_right = initial_right;
for iteration = 1:rounds
    left_history{iteration} = key_left;
    right_history{iteration} = key_right;
    key_left = circshift(key_left, 13);
    key_right = circshift(key_right, 17);
end

for iteration = rounds:-1:1
    key_left = left_history{iteration};
    key_right = right_history{iteration};

    injection_left = mod(left + key_left, 256);
    if right_length > left_length
        extended = zeros(right_length, 1);
        extended(1:left_length) = injection_left;
        extended(end) = injection_left(end);
        injection_left = extended;
    end
    right = mod(right - injection_left, 256);

    injection_right = mod(right + key_right, 256);
    left = mod(left - injection_right(1:left_length), 256);
end

output = uint8(reshape([left; right], h, w, d));
end
