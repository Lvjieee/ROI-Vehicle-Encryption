function init_global_pool(base_key, pool_size)
%INIT_GLOBAL_POOL Precompute a deterministic shared chaotic key pool.

validateattributes(pool_size, {'numeric'}, ...
    {'scalar', 'integer', 'positive', 'finite'});
validateattributes(base_key, {'numeric'}, ...
    {'vector', 'numel', 32, 'finite', 'real'});

base_state = 0.01 * ones(1, 6);
initial_state = generate_init(base_key, base_state);
step = 0.01;
discard = 5000;
sample_count = double(pool_size) + discard;

fprintf('[MATLAB] Generating key pool with %d rows...\n', pool_size);
[~, raw] = rossler(0:step:(sample_count - 1) * step, ...
    1e-5, 1e-5, initial_state);
key_pool_store('set', raw(discard + 1:end, :));
fprintf('[MATLAB] Key pool ready.\n');
end
