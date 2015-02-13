function test()
  I = im2double(imread('lena.png'));
  % I = I(:, 1:256);
  % I = checkerboard(50);
  [orig_h, orig_w] = size(I);

  % lambda  = 3;
  % theta   = 0;
  % psi     = [0 pi/2];
  % gamma   = 0.5;
  % bw      = 1;
  % N       = 10;
  % theta = theta - 2*pi/N;
  % filt = gabor_fn(bw,gamma,psi(1),lambda,theta) + 1i * gabor_fn(bw,gamma,psi(2),lambda,theta);
  % filt = fspecial('gaussian', [20 20], 5.5);
  % filt = fspecial('sobel');
  % filt = imresize(filt, 5);

  cano = real(conv2(I, filt, 'valid'));

  sfigure(1);
  subplot(221); imagesc(I); title('original');
  axis image off;
  % subplot(222); imagesc(cano); title('canonical output');
  subplot(222); imagesc(max(cano, 0)); title('canonical output');
  axis image off;
  subplot(224); imshow(filt, []); title('filter');
  colormap gray;
  [height, width] = size(cano);


  trans(1).scale = 2;
  trans(1).rotate = 0;

  trans(2).scale = 1.2;
  % trans(2).rotate = 10;
  trans(2).rotate = 0;

  trans(3).scale = 1;
  % trans(3).rotate = -5;
  trans(3).rotate = 0;

  trans(4).scale = 0.5;
  trans(4).rotate = 0;

  trans(5).scale = 1.;
  % trans(5).rotate = 5;
  trans(5).rotate = 0;

  nTrans = length(trans);

  sfigure(2); clf;
  for t = 1:nTrans
      I_here = I;
      % ----- forward phase ----- %
      if trans(t).scale ~= 1
          I_here = imresize(I_here, trans(t).scale, 'nearest');
      end
      if trans(t).rotate ~= 0
          I_here = imrotate(I_here, trans(t).rotate, 'nearest', 'loose');
      end

      % convolve image:
      subplot(3, nTrans, t);
      % sfigure(t); clf;
      % subplot(1, 3, 1);
      imagesc(I_here); axis image;
      title(sprintf('fwd: scale %.2f orientation %.2f', trans(t).scale, trans(t).rotate));
      I_here = real(conv2(I_here, filt, 'valid'));
      subplot(3, nTrans, t+nTrans);
      % subplot(1, 3, 2);
      imagesc(max(I_here, 0));  axis image;
      % imagesc(I_here);  axis image;
      title('after convolution');

      % ----- backward phase ----- %
      I_back = I_here;
      % undo transformation
      if trans(t).scale ~= 1
          scale_here = 1/trans(t).scale;
          I_back = imresize(I_back, scale_here, 'nearest');
      end
      if trans(t).rotate ~= 0
          I_back = imrotate(I_back, -trans(t).rotate, 'nearest', 'loose');
      end
      I_back  = crop_center(I_back, height, width);
      subplot(3, nTrans, 2*nTrans+t);
      % subplot(1, 3, 3);
      % imagesc(I_back); axis image;
      imagesc(max(I_back, 0)); axis image off;
      title(sprintf('bwd: scale %.2f orientation %.2f', 1/trans(t).scale, -trans(t).rotate));
      colormap gray;

      if size(I_back, 1) ~= size(cano, 1)
          fprintf('not same @ %d %d vs %d\n', t, size(I_back, 1), ...
                  size(cano, 1));
      end
      % assert(size(I_back, 2) == size(cano, 2));
      images{t} = I_back;
  end
  test = cat(3, images{:});
  sfigure(1);
  % subplot(223); imagesc(max(test, [], 3));
  subplot(223); imagesc(max(max(test, [], 3), 0));
  title('max response overall transformations');
  axis image off;
  keyboard


% function I_new = crop_center(I, height, width)
%     if size(I, 1) < height
%         I_new = zeros(height, width);
%         % offset = max(floor( (height - size(I, 1) -1 ) / 2), 1);
%         diff = (height - size(I, 1) );
%         offset = floor(diff / 2)+1;
%         I_new(offset:offset+size(I, 1) - 1, ...
%               offset:offset+size(I, 2) - 1) = I;
%     else
%         center = floor((size(I)-1)./2);
%         offset = floor((height-1)/2);
%         I_new = I(center(1)-offset:center(1)+offset+mod(height-1, 2), center(2)-offset:center(2)+offset+mod(height-1, 2));
%     end
%     assert(size(I_new, 1) == height);
%     assert(size(I_new, 1) == width);
    % function I_new = crop_left(I, height_orig, width_orig, height, width)
%   offset_r = floor((size(I, 1) - height_orig - 1)/2);
%   offset_c = floor((size(I, 2) - width_orig - 1)/2);
%   I_new = I(offset_r:offset_r+height-1, offset_c:offset_c+width-1);
%   assert(size(I_new, 1) == height);
%   assert(size(I_new, 1) == width);
