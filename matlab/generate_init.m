function x0 = generate_init(digest, key_c)
% Eqs. (2)-(3): big-endian 32-bit words of the digest scaled to [0,1],
% mixed, and added to the secret key c.
b = double(uint8(digest(:)'));
h = zeros(1, 8);
for i = 1:8
    p = 4*i - 3;
    h(i) = (b(p)*2^24 + b(p+1)*2^16 + b(p+2)*2^8 + b(p+3)) / (2^32 - 1);
end
mix = mod([h(1)+h(2), h(3)+h(4), h(5)+h(6), h(7)+h(8), ...
           ((h(1)+h(3))+h(5))+h(7), ((h(2)+h(4))+h(6))+h(8)], 1);
x0 = double(key_c(:)') + mix;
end
