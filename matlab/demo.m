% MATLAB demo: encrypt, decrypt, crop 25% and repair one ROI.
clc; clear; close all;
key_c = 0.01 * ones(1, 6);
T = 3;

if exist('sample_roi.png', 'file')
    roi = imread('sample_roi.png');
else
    [x, y] = meshgrid(uint8(0:255), uint8(0:255));
    roi = cat(3, x, y, bitxor(x, y));
end
roi = roi(:, :, 1:3);

tic; [cipher, digest] = encrypt(roi, key_c, T); te = toc;
tic; plain = decrypt(cipher, digest, key_c, T); td = toc;
assert(isequal(plain, roi));
fprintf('lossless, encrypt %.3f s, decrypt %.3f s\n', te, td);

[h, w, ~] = size(cipher);
dmg = cipher;
dmg(floor(h/4)+(1:floor(h/2)), floor(w/4)+(1:floor(w/2)), :) = 0;
raw = decrypt(dmg, digest, key_c, T, false);
rep = decrypt(dmg, digest, key_c, T, true);
fprintf('crop 25%%: PSNR %.2f dB -> %.2f dB with CCGR\n', ...
    psnr(rgb2gray(raw), rgb2gray(roi)), psnr(rgb2gray(rep), rgb2gray(roi)));

figure;
subplot(1, 4, 1); imshow(roi);    title('plain');
subplot(1, 4, 2); imshow(cipher); title('cipher');
subplot(1, 4, 3); imshow(dmg);    title('cropped');
subplot(1, 4, 4); imshow(rep);    title('CCGR');
