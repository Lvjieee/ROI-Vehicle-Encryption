function y = rossler(x0, n, step)
% n RK4 steps of the 6-D hyperchaotic system, Eq. (1).
if nargin < 3, step = 0.01; end
a = 36; b = 2.8; c = 28.6; d = 9.6; kappa = 1e-4;
f = @(s) [a*(s(2) - s(1)), c*s(2) - s(1)*s(3) + s(5), s(1)*s(2) - b*s(3), ...
          s(2)*s(3) - s(4), -kappa*s(2), s(1)*s(5) - d*s(6)];
y = zeros(n, 6);
x = double(x0(:)');
y(1, :) = x;
for i = 2:n
    k1 = f(x);
    k2 = f(x + 0.5*step*k1);
    k3 = f(x + 0.5*step*k2);
    k4 = f(x + step*k3);
    x = x + step/6 * (k1 + 2*k2 + 2*k3 + k4);
    if any(~(abs(x) <= 1e12))
        error('rossler:diverged', 'integration diverged at step %d', i - 1);
    end
    y(i, :) = x;
end
end
