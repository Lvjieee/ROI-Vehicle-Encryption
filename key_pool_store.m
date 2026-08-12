function [cipher_double, cipher_uint8, key_info, ...
    total_time, permutation_time, diffusion_time] = ...
    encrypt(image, rounds, start_index)
%ENCRYPT Encrypt one uint8 ROI.
%   SHA mode:
%     [~, C, DIGEST] = ENCRYPT(IMAGE, ROUNDS)
%   Shared-pool mode:
%     [~, C, START_INDEX] = ENCRYPT(IMAGE, ROUNDS, START_INDEX)
%
%   In shared-pool mode, KEY_INFO intentionally returns the index used for
%   this ROI, not the next free index. Store it with the ROI metadata.

validateattributes(image, {'uint8', 'numeric'}, ...
    {'nonempty', 'real', 'finite'});
validateattributes(rounds, {'numeric'}, ...
    {'scalar', 'integer', 'nonnegative', 'finite'});

image = uint8(image);
[h, w, d] = size(image);
permutation_rows = ceil((h * w * d) / 6);
required_length = pool_required_length(h, w, d);

total_timer = tic;
if nargin < 3
    digest = hash(image, 'SHA-256');
    base_state = 0.01 * ones(1, 6);
    initial_state = generate_init(digest, base_state);
    step = 0.01;
    discard = 5000;
    [~, all_keys] = rossler( ...
        0:step:(required_length + discard - 1) * step, ...
        1e-5, 1e-5, initial_state);
    key_slice = all_keys(discard + 1:end, :);
    key_info = double(digest);
else
    validateattributes(start_index, {'numeric'}, ...
        {'scalar', 'integer', 'positive', 'finite'});
    pool = key_pool_store('get');
    final_index = double(start_index) + required_length - 1;
    if final_index > size(pool, 1)
        error('encrypt:PoolRangeExceeded', ...
            'ROI needs pool rows %d:%d, but the pool has %d rows.', ...
            start_index, final_index, size(pool, 1));
    end
    key_slice = pool(double(start_index):final_index, :);
    key_info = double(start_index);
end

permutation_key = key_slice(1:permutation_rows, :);
diffusion_key = key_slice(permutation_rows+1:end, :);

timer = tic;
scrambled = permutation(image, permutation_key);
permutation_time = toc(timer);

timer = tic;
cipher_uint8 = myDiffusion(scrambled, diffusion_key, rounds);
diffusion_time = toc(timer);

cipher_double = double(cipher_uint8);
total_time = toc(total_timer);
end
