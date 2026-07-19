function initial_state = generate_init(key_bytes, base_state)
%GENERATE_INIT Deterministically map a 256-bit key to six finite states.
%   This implementation deliberately avoids interpreting arbitrary digest
%   bytes as IEEE-754 floats, which can produce NaN or Inf.

validateattributes(key_bytes, {'numeric'}, ...
    {'vector', 'numel', 32, 'finite', 'real'});
validateattributes(base_state, {'numeric'}, ...
    {'vector', 'numel', 6, 'finite', 'real'});

key_bytes = uint8(mod(floor(double(key_bytes(:)')), 256));
base_state = double(base_state(:)');

words = zeros(1, 8);
for i = 1:8
    p = 4 * (i - 1) + 1;
    b = double(key_bytes(p:p+3));
    % Explicit big-endian conversion makes the mapping platform-independent.
    words(i) = b(1) * 2^24 + b(2) * 2^16 + b(3) * 2^8 + b(4);
end
u = words / (2^32 - 1);

mix = [ ...
    mod(u(1) + u(2), 1), ...
    mod(u(3) + u(4), 1), ...
    mod(u(5) + u(6), 1), ...
    mod(u(7) + u(8), 1), ...
    mod(sum(u(1:2:8)), 1), ...
    mod(sum(u(2:2:8)), 1) ...
];

initial_state = base_state + mix;
if any(~isfinite(initial_state))
    error('generate_init:NonFiniteState', ...
        'Key mapping produced a non-finite initial state.');
end
end
