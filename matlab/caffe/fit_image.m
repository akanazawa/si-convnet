function I_adj = fit_image(I, img_size, pad_value)
 if nargin == 2
     pad_value = 0;
 end

 new_r = size(I, 1);
 new_c = size(I, 2);
 
 if new_r ~= new_c
     if new_r < new_c
         I = crop_or_center_image(I, new_r, pad_value);         
     else
         I = crop_or_center_image(I, new_c, pad_value);         
     end     
 end
 
 I_adj = crop_or_center_image(I, img_size, pad_value);         

function I_adj = crop_or_center_image(I, img_size, pad_value) 
 
 new_r = size(I, 1);
 new_c = size(I, 2);
 if new_r < img_size || new_c < img_size
     I_adj = pad_value*ones(img_size, img_size);
     diff_r = img_size-new_r;
     diff_c = img_size-new_c;
     offset_r = floor(diff_r / 2) + 1;
     offset_c = floor(diff_c / 2) + 1;     
     I_adj(offset_r:offset_r+new_r - 1, offset_c:offset_c+new_c - 1) = ...
         I;
 else        
     if (new_r == img_size) && (new_c == img_size)
         I_adj = I;
     else            
         c_r = floor((new_r - 1)/2) + 1;
         c_c = floor((new_c - 1)/2) + 1;
         offset = floor((img_size - 1)/2);
         I_adj = I(c_r-offset:c_r+offset + ~mod(img_size, 2), ...
                   c_c-offset:c_c+offset + ~mod(img_size, 2));
     end
 end
 assert(size(I_adj, 1) == img_size);
