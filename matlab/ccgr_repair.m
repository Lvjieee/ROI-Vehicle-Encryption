function out = ccgr_repair(img, mask)
% CCGR: per-channel regionfill, then I_c = fill(I_c - I_r) + I_r averaged
% over the reference channels r that are intact, Eq. (9).
[h, w, d] = size(img);
x = double(img);
for c = 1:d
    if any(any(mask(:, :, c)))
        x(:, :, c) = regionfill(x(:, :, c), mask(:, :, c));
    end
end
if d == 3
    for c = 1:3
        mc = mask(:, :, c);
        if ~any(mc(:)), continue; end
        total = zeros(h, w); count = zeros(h, w);
        for r = setdiff(1:3, c)
            t = mc & ~mask(:, :, r);
            if ~any(t(:)), continue; end
            diff = regionfill(x(:, :, c) - x(:, :, r), mc | mask(:, :, r));
            ref = x(:, :, r);
            total(t) = total(t) + ref(t) + diff(t);
            count(t) = count(t) + 1;
        end
        ch = x(:, :, c);
        ch(count > 0) = total(count > 0) ./ count(count > 0);
        x(:, :, c) = ch;
    end
end
out = uint8(x);
end
