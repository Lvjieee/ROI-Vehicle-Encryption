function required_length = pool_required_length(height, width, depth)
%POOL_REQUIRED_LENGTH Number of six-value chaotic rows needed for one ROI.

validateattributes(height, {'numeric'}, {'scalar', 'integer', 'positive'});
validateattributes(width, {'numeric'}, {'scalar', 'integer', 'positive'});
validateattributes(depth, {'numeric'}, {'scalar', 'integer', 'positive'});

permutation_rows = ceil((height * width * depth) / 6);
diffusion_rows = ceil((height * width) / 2);
required_length = permutation_rows + diffusion_rows;
end
