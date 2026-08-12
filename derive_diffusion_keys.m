function digest = hash(input_value, method)
%HASH Return a byte-vector message digest.
%   DIGEST = HASH(INPUT_VALUE, 'SHA-256') hashes the exact byte
%   representation of INPUT_VALUE. For image data, callers should pass a
%   uint8 array so the digest is independent of MATLAB's floating-point
%   representation.

if nargin < 2
    method = 'SHA-256';
end

method = upper(strrep(char(method), '_', '-'));
if strcmp(method, 'SHA256')
    method = 'SHA-256';
end
if ~strcmp(method, 'SHA-256')
    error('hash:UnsupportedMethod', ...
        'This research implementation supports SHA-256 only.');
end

if isa(input_value, 'uint8')
    bytes = input_value(:);
elseif ischar(input_value) || isstring(input_value)
    bytes = unicode2native(char(input_value), 'UTF-8');
    bytes = uint8(bytes(:));
else
    bytes = typecast(input_value(:), 'uint8');
end

engine = java.security.MessageDigest.getInstance('SHA-256');
engine.update(bytes);
digest = reshape(typecast(engine.digest(), 'uint8'), 1, []);
end
