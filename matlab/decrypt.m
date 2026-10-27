function plain = decrypt(cipher, digest, key_c, rounds, recover)
% plain = decrypt(cipher, digest, key_c, rounds, recover)
% recover = true runs CCGR when the damage test, Eq. (8), fires
% (needs regionfill, Image Processing Toolbox).
if nargin < 4, rounds = 3; end
if nargin < 5, recover = false; end
cipher = uint8(cipher);
[h, w, d] = size(cipher);
[K1, K2] = keystreams(digest, key_c, h, w, d);
plain = uint8(de_permutation(myDeDiffusion(cipher, K2, rounds), K1));
bad = cipher == 0 | cipher == 255;
if recover && nnz(bad) > 1.5 * numel(cipher) * 2/256
    plain = ccgr_repair(plain, ccgr_mask(bad, K1, rounds));
end
end
