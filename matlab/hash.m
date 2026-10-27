function digest = hash(x, ~)
% SHA-256 of a uint8 array in column-major order (Java MessageDigest).
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(uint8(x(:)));
digest = reshape(typecast(md.digest(), 'uint8'), 1, []);
end
