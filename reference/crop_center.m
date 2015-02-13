function I_new = crop_center(I, height, width)
% Crops the image I to height and width at the center. If the new
% dimensions are odd, extra pixel happens to the right of the center.
I_new = zeros(height, width);
    % Get where nonzero rows start in I_new.
    new_c_start = 0;
    new_c_end = width - 1;
    new_r_start = 0;
    new_r_end = height - 1;
    old_r_start = 0;
    old_r_end = size(I, 1) - 1;
    old_c_start = 0;
    old_c_end = size(I, 2) - 1;
    if size(I, 1) < height % new bigger, pad
        center_r = height / 2;
        offset = size(I,1) / 2;
        new_r_start = ceil(center_r - offset);
        new_r_end = ceil(center_r + offset) - 1;
    else % Crop center
        center_r = size(I, 1) / 2;
        offset = height/2;
        old_r_start = ceil(center_r - offset);
        old_r_end = ceil(center_r + offset) - 1;
    end
    if size(I, 2) < width
        center_c = width / 2;
        offset  = size(I,2) / 2;
        new_c_start = ceil(center_c - offset);
        new_c_end = ceil(center_c + offset) - 1;
    else
        center_c = size(I, 2) / 2;
        offset = width / 2;
        old_c_start = ceil(center_c - offset);
        old_c_end = ceil(center_c + offset) - 1;
    end
    % Bc matlab counts from 1.
    old_r_start = old_r_start + 1;
    new_r_start = new_r_start + 1;
    old_c_start = old_c_start + 1;
    new_c_start = new_c_start + 1;
    old_r_end = old_r_end + 1;
    new_r_end = new_r_end + 1;
    old_c_end = old_c_end + 1;
    new_c_end = new_c_end + 1;
    I_new(new_r_start:new_r_end, new_c_start:new_c_end) = I(old_r_start:old_r_end, old_c_start:old_c_end);
    orig_inds = reshape([0:(numel(I)-1)], size(I, 2), size(I, 1))';
    new_inds = -ones(height, width);
    new_inds(new_r_start:new_r_end, new_c_start:new_c_end) = ...
        orig_inds(old_r_start:old_r_end, old_c_start:old_c_end);
    for i = 1:size(new_inds, 1)
        for j = 1:size(new_inds, 2)
            fprintf('%3d,', new_inds(i,j));
            if (j ~= size(new_inds, 2))
                fprintf(' ');
            end
        end
        fprintf('\n');
    end
    keyboard
