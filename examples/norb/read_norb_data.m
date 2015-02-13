function read_norb_data()

dir = '/scratch0/data/norb/';
% label_name = fullfile(dir, 'norb-5x01235x9x18x6x2x108x108-testing-01-cat.mat');
% data_name = fullfile(dir, 'norb-5x01235x9x18x6x2x108x108-testing-01-dat.mat');
%label_name = fullfile(dir, 'norb-5x46789x9x18x6x2x108x108-training-10-cat.mat');
%data_name = fullfile(dir, 'norb-5x46789x9x18x6x2x108x108-training-10-dat.mat');
label_name = fullfile(dir, 'norb-5x46789x9x18x6x2x108x108-training-01-cat.mat');
data_name = fullfile(dir, 'norb-5x46789x9x18x6x2x108x108-training-01-dat.mat');

fname = data_name;

images = read_file(data_name);
labels = read_file(label_name);

keyboard
function result = read_file(fname) 

fid = fopen(fname);

magic = fread(fid, 4, 'uchar');

if all(magic == [85 76 61 30]') % byte
    type = 'uint8=>uint8';
elseif all(magic == [84 76 61 30]') % int
    type = 'int';
elseif all(magic == [86 76 61 30]') % short
    type = 'uint16';
elseif all(magic == [83 76 61 30]') % double
    type = 'double';
elseif all(magic == [82 76 61 30]') % single
    type = 'single';
elseif all(magic == [81 76 61 30]') % packed
    fprintf('packed matrix.. what to use..?\n');
    type = 'single';
else
    fprintf('unknown magic\n');
    keyboard;
end
fprintf('magic %s\n', type);

ndim = fread(fid, 4, 'uchar'); 
ndim = ndim(1);
fprintf('ndim %d\n', ndim);

dim0 = fread(fid, 4, 'uchar'); 
dim0 = sum(dim0'.*(256.^[0:3]))

dim1 = fread(fid, 4, 'uchar'); 
dim1 = dim1(1);

dim2 = fread(fid, 4, 'uchar'); 
dim2 = dim2(1);

if ndim == 4
    dim3 = fread(fid, 4, 'uchar'); 
    dim3 = dim3(1);
end

if ndim == 4
    dim0 = 100;
    result = zeros(dim2, dim3, dim1, dim0, 'uint8');
    for i = 1:dim0
        data0 = fread(fid, dim2*dim3, type);
        data1 = fread(fid, dim2*dim3, type);
        sfigure(1); 
        subplot(121);imshow(reshape(data0, dim2, dim3)', []);
        subplot(122);imshow(reshape(data1, dim2, dim3)', []);
        result(:, :, 1, i) = reshape(data0, dim2, dim3)';
        result(:, :, 2, i) = reshape(data1, dim2, dim3)';
        keyboard
    end
else ndim == 1    
    result = fread(fid, dim0, type);
end

fclose(fid);

