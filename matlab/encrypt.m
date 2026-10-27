function [cipher, digest] = encrypt(roi, key_c, rounds)
% [cipher, digest] = encrypt(roi, key_c, rounds)
% roi: H x W x 3 uint8, key_c: secret key c (6 numbers), rounds: T (3).
% digest is sent with the ciphertext; it is not secret.
if nargin < 3, rounds = 3; end
roi = uint8(roi);
[h, w, d] = size(roi);
digest = hash(roi);
[K1, K2] = keystreams(digest, key_c, h, w, d);
cipher = myDiffusion(permutation(roi, K1), K2, rounds);
end
