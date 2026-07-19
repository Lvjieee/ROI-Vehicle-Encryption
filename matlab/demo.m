clc;
clear;
close all;

% A generated sample keeps the demo self-contained.
[x, y] = meshgrid(uint8(0:255), uint8(0:255));
plain = cat(3, x, y, bitxor(x, y));
rounds = 3;

[~, cipher, digest, total_time, permutation_time, diffusion_time] = ...
    encrypt(plain, rounds);
recovered = decrypt(cipher, digest, rounds, false);

assert(isequal(plain, recovered), 'Round-trip recovery was not lossless.');
fprintf('Lossless SHA-mode round trip passed.\n');
fprintf('Total %.4f s | permutation %.4f s | diffusion %.4f s\n', ...
    total_time, permutation_time, diffusion_time);

figure('Name', 'ROI encryption demo');
subplot(1, 3, 1); imshow(plain); title('Plain ROI');
subplot(1, 3, 2); imshow(cipher); title('Cipher ROI');
subplot(1, 3, 3); imshow(recovered); title('Recovered ROI');
