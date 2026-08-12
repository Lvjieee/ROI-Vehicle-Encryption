function result = key_pool_store(action, value)
%KEY_POOL_STORE Controlled persistent storage for the shared chaotic pool.

persistent pool;
if nargin < 1
    error('key_pool_store:MissingAction', 'An action is required.');
end

action = char(action);
if strcmpi(action, 'set')
        if nargin < 2
            error('key_pool_store:MissingValue', ...
                'The set action requires a pool value.');
        end
        validateattributes(value, {'numeric'}, ...
            {'2d', 'nonempty', 'finite', 'real'});
        if size(value, 2) ~= 6
            error('key_pool_store:InvalidShape', ...
                'The key pool must have six columns.');
        end
        pool = double(value);
        result = size(pool, 1);
elseif strcmpi(action, 'get')
        if isempty(pool)
            error('key_pool_store:NotInitialized', ...
                'Call init_global_pool before pool-mode encryption.');
        end
        result = pool;
elseif strcmpi(action, 'clear')
        pool = [];
        result = 0;
else
        error('key_pool_store:UnknownAction', ...
            'Unknown key-pool action: %s.', action);
end
end
