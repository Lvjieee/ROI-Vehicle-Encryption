function [t, y] = rossler(tspan, ~, ~, x0)
%ROSSLER Deterministic fixed-step integration of the 6-D chaotic system.
%   Numerical failure is reported instead of being repaired with random
%   noise, because encryption and decryption must reproduce identical
%   trajectories in independent processes.

validateattributes(tspan, {'numeric'}, ...
    {'vector', 'real', 'finite', 'increasing'});
validateattributes(x0, {'numeric'}, ...
    {'vector', 'numel', 6, 'real', 'finite'});
if numel(tspan) < 2
    error('rossler:ShortTimeGrid', ...
        'tspan must contain at least two samples.');
end

a = 36;
b = 2.8;
c = 28.6;
d = 9.6;
coupling = 0.0001;

t = double(tspan(:));
count = length(t);
step = t(2) - t(1);
grid_tolerance = 1e-9 * max(1, abs(step));
if any(abs(diff(t) - step) > grid_tolerance)
    error('rossler:NonUniformGrid', 'tspan must use a fixed step.');
end

y = zeros(count, 6);
y(1, :) = double(x0(:)');
limit = 1e12;

for i = 1:count-1
    state = y(i, :);
    k1 = derivative(state);
    k2 = derivative(state + 0.5 * step * k1);
    k3 = derivative(state + 0.5 * step * k2);
    k4 = derivative(state + step * k3);
    next_state = state + step / 6 * (k1 + 2*k2 + 2*k3 + k4);

    if any(~isfinite(next_state)) || any(abs(next_state) > limit)
        error('rossler:IntegrationDiverged', ...
            'Chaotic integration diverged at step %d.', i);
    end
    y(i+1, :) = next_state;
end

    function dx = derivative(state)
        dx = zeros(1, 6);
        dx(1) = a * (state(2) - state(1));
        dx(2) = c * state(2) - state(1) * state(3) + state(5);
        dx(3) = state(1) * state(2) - b * state(3);
        dx(4) = state(2) * state(3) - state(4);
        dx(5) = -coupling * state(2);
        dx(6) = state(1) * state(5) - d * state(6);
    end
end
