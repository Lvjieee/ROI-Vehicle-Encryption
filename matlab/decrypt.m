function output = decrypt(cipher_image, key_info, rounds, enable_recovery)
%DECRYPT Decrypt one ROI in SHA or shared-pool mode.
%   Set ENABLE_RECOVERY=true only when the ciphertext is known to have
%   suffered zeroing or salt-and-pepper damage. Standard video decryption
%   should leave it false to preserve exact pixels.

if nargin < 4
    enable_recovery = false;
end
validateattributes(cipher_image, {'uint8', 'numeric'}, ...
    {'nonempty', 'real', 'finite'});
validateattributes(rounds, {'numeric'}, ...
    {'scalar', 'integer', 'nonnegative', 'finite'});
validateattributes(enable_recovery, {'logical', 'numeric'}, {'scalar'});

cipher_image = uint8(cipher_image);
[h, w, d] = size(cipher_image);
permutation_rows = ceil((h * w * d) / 6);
required_length = pool_required_length(h, w, d);

if numel(key_info) == 32
    base_state = 0.01 * ones(1, 6);
    initial_state = generate_init(key_info, base_state);
    step = 0.01;
    discard = 5000;
    [~, all_keys] = rossler( ...
        0:step:(required_length + discard - 1) * step, ...
        1e-5, 1e-5, initial_state);
    key_slice = all_keys(discard + 1:end, :);
elseif isscalar(key_info)
    validateattributes(key_info, {'numeric'}, ...
        {'scalar', 'integer', 'positive', 'finite'});
    pool = key_pool_store('get');
    final_index = double(key_info) + required_length - 1;
    if final_index > size(pool, 1)
        error('decrypt:PoolRangeExceeded', ...
            'ROI needs pool rows %d:%d, but the pool has %d rows.', ...
            key_info, final_index, size(pool, 1));
    end
    key_slice = pool(double(key_info):final_index, :);
else
    error('decrypt:InvalidKeyInfo', ...
        'key_info must be a 32-byte digest or a scalar pool index.');
end

permutation_key = key_slice(1:permutation_rows, :);
diffusion_key = key_slice(permutation_rows+1:end, :);

attack_mask = false(h, w, d);
if logical(enable_recovery)
    total_pixels = numel(cipher_image);
    expected_extreme = total_pixels * (2 / 256);
    extreme_mask = cipher_image == 0 | cipher_image == 255;
    is_attacked = nnz(extreme_mask) > 1.5 * expected_extreme;

    if is_attacked
        flat = extreme_mask(:);
        left_length = floor(total_pixels / 2);
        right_length = total_pixels - left_length;
        mask_left = flat(1:left_length);
        mask_right = flat(left_length+1:end);
        for iteration = rounds:-1:1
            propagated_left = mask_left;
            if right_length > left_length
                extended = false(right_length, 1);
                extended(1:left_length) = propagated_left;
                extended(end) = propagated_left(end);
                propagated_left = extended;
            end
            mask_right = mask_right | propagated_left;
            mask_left = mask_left | mask_right(1:left_length);
        end
        permuted_mask = reshape([mask_left; mask_right], h, w, d);
        attack_mask = de_permutation(double(permuted_mask), ...
            permutation_key) > 0;
    end
else
    is_attacked = false;
end

undiffused = myDeDiffusion(cipher_image, diffusion_key, rounds);
output = uint8(mod(de_permutation(undiffused, permutation_key), 256));

if is_attacked
    if exist('regionfill', 'file') ~= 2
        error('decrypt:MissingRegionfill', ...
            ['Recovery requires regionfill from the MATLAB Image ', ...
             'Processing Toolbox.']);
    end
    output = repair_channels(output, attack_mask);
end
end

function repaired = repair_channels(image, attack_mask)
[h, w, d] = size(image);
working = double(image);

for channel = 1:d
    mask = attack_mask(:, :, channel);
    if any(mask(:))
        working(:, :, channel) = regionfill(working(:, :, channel), mask);
    end
end

if d == 3
    for channel = 1:d
        target_mask = attack_mask(:, :, channel);
        if ~any(target_mask(:))
            continue;
        end
        prediction_sum = zeros(h, w);
        prediction_count = zeros(h, w);
        for reference = setdiff(1:d, channel)
            reference_mask = attack_mask(:, :, reference);
            valid_target = target_mask & ~reference_mask;
            if ~any(valid_target(:))
                continue;
            end
            difference = working(:, :, channel) - working(:, :, reference);
            missing = target_mask | reference_mask;
            filled_difference = regionfill(difference, missing);
            reference_data = working(:, :, reference);
            prediction_sum(valid_target) = prediction_sum(valid_target) + ...
                reference_data(valid_target) + ...
                filled_difference(valid_target);
            prediction_count(valid_target) = ...
                prediction_count(valid_target) + 1;
        end
        valid = prediction_count > 0;
        channel_data = working(:, :, channel);
        channel_data(valid) = prediction_sum(valid) ./ prediction_count(valid);
        working(:, :, channel) = channel_data;
    end
end

repaired = uint8(min(max(round(working), 0), 255));
end
