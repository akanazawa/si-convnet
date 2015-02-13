"""
Do windowed detection by classifying a number of images/crops at once,
optionally using the selective search window proposal method.

This implementation follows
  Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik.
  Rich feature hierarchies for accurate object detection and semantic
  segmentation.
  http://arxiv.org/abs/1311.2524

The selective_search_ijcv_with_python code is available at
  https://github.com/sergeyk/selective_search_ijcv_with_python

TODO:
- batch up image filenames as well: don't want to load all of them into memory
- refactor into class (without globals)
- update demo notebook with new options
"""
import numpy as np
import os
import sys
import gflags
import pandas as pd
import time
import skimage.io
import skimage.transform
import selective_search_ijcv_with_python as selective_search
import caffe
import re
import scipy.io

NET = None

IMAGE_DIM = None
CROPPED_DIM = None
IMAGE_CENTER = None

IMAGE_MEAN = None
CROPPED_IMAGE_MEAN = None

BATCH_SIZE = None
NUM_OUTPUT = None

CROP_MODES = ['list', 'center_only', 'corners', 'selective_search']

def load_image(filename):
  """
  Input:
    filename: string

  Output:
    image: an image of size (H x W x 3) of type uint8.
  """
  img = skimage.io.imread(filename)
  if img.ndim == 2:
    img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
  elif img.shape[2] == 4:
    img = img[:, :, :3]
  return img


def format_image(image, window=None, cropped_size=False):
  """
  Input:
    image: (H x W x 3) ndarray
    window: (4) ndarray
      (ymin, xmin, ymax, xmax) coordinates, 0-indexed
    cropped_size: bool
      Whether to output cropped size image or full size image.

  Output:
    image: (3 x H x W) ndarray
      Resized to either IMAGE_DIM or CROPPED_DIM.
    dims: (H, W) of the original image
  """
  dims = image.shape[:2]

  # Crop a subimage if window is provided.
  if window is not None:
    image = image[window[0]:window[2], window[1]:window[3]]

  # Resize to input size, subtract mean, convert to BGR
  image = image[:, :, ::-1]
  if cropped_size:
    image = skimage.transform.resize(image, (CROPPED_DIM, CROPPED_DIM)) * 255
    image -= CROPPED_IMAGE_MEAN
  else:
    image = skimage.transform.resize(image, (IMAGE_DIM, IMAGE_DIM)) * 255
    image -= IMAGE_MEAN

  image = image.swapaxes(1, 2).swapaxes(0, 1)
  return image, dims


def _image_coordinates(dims, window):
  """
  Calculate the original image coordinates of a
  window in the canonical (IMAGE_DIM x IMAGE_DIM) coordinates

  Input:
    dims: (H, W) of the original image
    window: (ymin, xmin, ymax, xmax) in the (IMAGE_DIM x IMAGE_DIM) frame

  Output:
    image_window: (ymin, xmin, ymax, xmax) in the original image frame
  """
  h, w = dims
  h_scale, w_scale = h / IMAGE_DIM, w / IMAGE_DIM
  image_window = window * np.array((1. / h_scale, 1. / w_scale,
                                   h_scale, w_scale))
  return image_window.round().astype(int)


def _assemble_images_list(input_df):
  """
  For each image, collect the crops for the given windows.

  Input:
    input_df: pandas.DataFrame
      with 'filename', 'ymin', 'xmin', 'ymax', 'xmax' columns

  Output:
    images_df: pandas.DataFrame
      with 'image', 'window', 'filename' columns
  """
  # unpack sequence of (image filename, windows)
  windows = input_df[['ymin', 'xmin', 'ymax', 'xmax']].values
  image_windows = (
    (ix, windows[input_df.index.get_loc(ix)]) for ix in input_df.index.unique()
  )

  # extract windows
  data = []
  for image_fname, windows_here in image_windows:
    image = load_image(image_fname)
    # AJ: Need to make it into a row vec if there's only 1 window
    if len(windows_here.shape) == 1:
      windows_here = np.reshape(windows_here, (1, 4))
    for window_here in windows_here:
      window_image, _ = format_image(image, window_here, cropped_size=True)
      data.append({
        'image': window_image[np.newaxis, :],
        'window': window_here,
        'filename': image_fname
      })

  images_df = pd.DataFrame(data)
  return images_df


def _assemble_images_center_only(image_fnames):
  """
  For each image, square the image and crop its center.

  Input:
    image_fnames: list

  Output:
    images_df: pandas.DataFrame
      With 'image', 'window', 'filename' columns.
  """
  crop_start, crop_end = IMAGE_CENTER, IMAGE_CENTER + CROPPED_DIM
  crop_window = np.array((crop_start, crop_start, crop_end, crop_end))

  data = []
  for image_fname in image_fnames:
    image, dims = format_image(load_image(image_fname))
    data.append({
      'image': image[np.newaxis, :,
                     crop_start:crop_end,
                     crop_start:crop_end],
      'window': _image_coordinates(dims, crop_window),
      'filename': image_fname
    })

  images_df = pd.DataFrame(data)
  return images_df

def _assemble_images_corners(image_fnames):
  """
  For each image, square the image and crop its center, four corners,
  and mirrored version of the above.

  Input:
    image_fnames: list

  Output:
    images_df: pandas.DataFrame
      With 'image', 'window', 'filename' columns.
  """
  # make crops
  indices = [0, IMAGE_DIM - CROPPED_DIM]
  crops = np.empty((5, 4), dtype=int)
  curr = 0
  for i in indices:
    for j in indices:
      crops[curr] = (i, j, i + CROPPED_DIM, j + CROPPED_DIM)
      curr += 1
  crops[4] = (IMAGE_CENTER, IMAGE_CENTER,
              IMAGE_CENTER + CROPPED_DIM, IMAGE_CENTER + CROPPED_DIM)
  all_crops = np.tile(crops, (2, 1))

  data = []
  for image_fname in image_fnames:
    image, dims = format_image(load_image(image_fname))
    image_crops = np.empty((10, 3, CROPPED_DIM, CROPPED_DIM), dtype=np.float32)
    curr = 0
    for crop in crops:
      image_crops[curr] = image[:, crop[0]:crop[2], crop[1]:crop[3]]
      curr += 1
    image_crops[5:] = image_crops[:5, :, :, ::-1]  # flip for mirrors
    for i in range(len(all_crops)):
      data.append({
        'image': image_crops[i][np.newaxis, :],
        'window': _image_coordinates(dims, all_crops[i]),
        'filename': image_fname
      })

  images_df = pd.DataFrame(data)
  return images_df


def _assemble_images_selective_search(image_fnames, n_max=-1):
  """
  Run Selective Search window proposals on all images, then for each
  image-window pair, extract a square crop.

  Input:
    image_fnames: list
    n_max: max number of boxes to mine, -1 means no restriction

  Output:
    images_df: pandas.DataFrame
      With 'image', 'window', 'filename' columns.
  """
  windows_list = selective_search.get_windows(image_fnames, n_max)
  data = []
  nWindows = 0.
  for image_fname, windows in zip(image_fnames, windows_list):
    image = load_image(image_fname)
    print "{}: {:d} windows found".format(image_fname, windows.shape[0])
    nWindows += windows.shape[0]
    for window in windows:
      window_image, _ = format_image(image, window, cropped_size=True)
      data.append({
        'image': window_image[np.newaxis, :],
        'window': window,
        'filename': image_fname
      })
      
  nImgs = len(image_fnames)    
  print "mean number of windows found for {:d} images : {:.3f}".format(nImgs, nWindows/nImgs)
  images_df = pd.DataFrame(data)
  return images_df


def assemble_batches(inputs, crop_mode='center_only', n_max=-1):
  """
  Assemble DataFrame of image crops for feature computation.

  Input:
    inputs: list of filenames (center_only, corners, and selective_search mode)
      OR input DataFrame (list mode)
    mode: string
      'list': take the image windows from the input as-is
      'center_only': take the CROPPED_DIM middle of the image windows
      'corners': take CROPPED_DIM-sized boxes at 4 corners and center of
        the image windows, as well as their flipped versions: a total of 10.
      'selective_search': run Selective Search region proposal on the
        image windows, and take each enclosing subwindow.

  Output:
    df_batches: list of DataFrames, each one of BATCH_SIZE rows.
      Each row has 'image', 'filename', and 'window' info.
      Column 'image' contains (X x 3 x CROPPED_DIM x CROPPED_IM) ndarrays.
      Column 'filename' contains source filenames.
      Column 'window' contains [ymin, xmin, ymax, xmax] ndarrays.
      If 'filename' is None, then the row is just for padding.

  Note: for increased efficiency, increase the batch size (to the limit of gpu
  memory) to avoid the communication cost
  """
  if crop_mode == 'list':
    images_df = _assemble_images_list(inputs)

  elif crop_mode == 'center_only':
    images_df = _assemble_images_center_only(inputs)

  elif crop_mode == 'corners':
    images_df = _assemble_images_corners(inputs)

  elif crop_mode == 'selective_search':
    images_df = _assemble_images_selective_search(inputs, n_max)

  else:
    raise Exception("Unknown mode: not in {}".format(CROP_MODES))

  # Make sure the DataFrame has a multiple of BATCH_SIZE rows:
  # just fill the extra rows with NaN filenames and all-zero images.
  N = images_df.shape[0]
  remainder = N % BATCH_SIZE
  if remainder > 0:
    zero_image = np.zeros_like(images_df['image'].iloc[0])
    zero_window = np.zeros((1, 4), dtype=int)
    remainder_df = pd.DataFrame([{
      'filename': None,
      'image': zero_image,
      'window': zero_window
    }] * (BATCH_SIZE - remainder))
    images_df = images_df.append(remainder_df)
    N = images_df.shape[0]

  # Split into batches of BATCH_SIZE.
  ind = np.arange(N) / BATCH_SIZE
  df_batches = [images_df[ind == i] for i in range(N / BATCH_SIZE)]
  return df_batches


def compute_feats(images_df):
  input_blobs = [np.ascontiguousarray(
    np.concatenate(images_df['image'].values), dtype='float32')]
  output_blobs = [np.empty((BATCH_SIZE, NUM_OUTPUT, 1, 1), dtype=np.float32)]

  NET.Forward(input_blobs, output_blobs)
  feats = [output_blobs[0][i].flatten() for i in range(len(output_blobs[0]))]
  image = np.concatenate(images_df['image'].values)
  scipy.io.savemat('imgandfeat000005.mat', dict(feats=feats, image=image))
  import ipdb as pdb; pdb.set_trace()
  # Add the features and delete the images.
  del images_df['image']
  images_df['feat'] = feats
  return images_df

def save_each_image(df, FLAGS):
  # coord_cols = ['ymin', 'xmin', 'ymax', 'xmax']
  # df[coord_cols] = pd.DataFrame(data=np.vstack(df['window']),
  #                             index=df.index,columns=coord_cols)
  # del df['window']
  # get unique fnames
  
  pattern = '(?<={})(.*)(?=\.(jpg|png|tiff|JPG|JPEG|PNG))'.format(\
      FLAGS.image_dir if FLAGS.image_dir[-1]=='/' else FLAGS.image_dir+"/")
  def get_bare_name(pattern, fname):
      match = re.search(pattern, fname)
      return match.group() if match is not None else fname

  fnames = set(df.filename.values.flat)
  for fname in fnames:
    df_here = df[df.filename == fname]
    fout = os.path.join(FLAGS.output_dir, '{}{}'.format(get_bare_name(pattern, fname), FLAGS.out_file_suffix))    
    if FLAGS.out_file_suffix.lower().endswith('.mat'):
      feat = np.asarray([_ for _ in df_here['feat'].as_matrix()])
      window = np.asarray([_ for _ in df_here['window'].as_matrix()], dtype=np.uint16)
      scipy.io.savemat(fout, dict(feat=feat, window=window))
    elif FLAGS.out_file_suffix.lower().endswith('.h5'):
      # del df_here['filename']
      # df_here.to_hdf(fout, 'df', mode='w')
      # matlab can't read these well
      store = pd.HDFStore(fout)
      store['feat'] = df_here['feat']
      store['window'] = df_here['window']
    else:
      print('undefined output file format')
      import ipdb as pdb; pdb.set_trace()
  
def save_results(df, FLAGS):
  # --------------------
  # Write out the results.
  # --------------------
  # for each image separately
  if FLAGS.save_each_image is True:
    save_each_image(df, FLAGS)
  else:
    df.set_index('filename', inplace=True)
    # print("Processing complete after {:.3f} s.".format(time.time() - t))

    if FLAGS.output_file.lower().endswith('csv'):
      # enumerate the class probabilities
      class_cols = ['class{}'.format(x) for x in range(NUM_OUTPUT)]
      df[class_cols] = pd.DataFrame(data=np.vstack(df['feat']),
                                    index=df.index,
                                    columns=class_cols)
      df.to_csv(FLAGS.output_file, sep=',',
                cols=coord_cols + class_cols,
                header=True)
    elif FLAGS.output_file.lower().endswith('mat'):
      feat = np.asarray([_ for _ in df['feat'].as_matrix()])
      window = np.asarray([_ for _ in df['window'].as_matrix()], dtype=np.uint16)
      fname = df.index.values
      scipy.io.savemat(FLAGS.output_file, dict(feat=feat, window=window, fname=fname))
    else:
      # Label coordinates
      coord_cols = ['ymin', 'xmin', 'ymax', 'xmax']
      df[coord_cols] = pd.DataFrame(data=np.vstack(df['window']),
                                    index=df.index,
                                    columns=coord_cols)
      del(df['window'])

      df.to_hdf(FLAGS.output_file, 'df', mode='w')



  
def config(model_def, pretrained_model, gpu, image_dim, image_mean_file):
  global IMAGE_DIM, CROPPED_DIM, IMAGE_CENTER, IMAGE_MEAN, CROPPED_IMAGE_MEAN
  global NET, BATCH_SIZE, NUM_OUTPUT

  # Initialize network by loading model definition and weights.
  t = time.time()
  print("Loading Caffe model.")
  NET = caffe.CaffeNet(model_def, pretrained_model)
  NET.set_phase_test()
  if gpu:
    NET.set_mode_gpu()
  print("Caffe model loaded in {:.3f} s".format(time.time() - t))

  # Configure for input/output data
  IMAGE_DIM = image_dim
  CROPPED_DIM = NET.blobs()[0].width
  IMAGE_CENTER = int((IMAGE_DIM - CROPPED_DIM) / 2)

    # Load the data set mean file
  IMAGE_MEAN = np.load(image_mean_file)

  CROPPED_IMAGE_MEAN = IMAGE_MEAN[IMAGE_CENTER:IMAGE_CENTER + CROPPED_DIM,
                                  IMAGE_CENTER:IMAGE_CENTER + CROPPED_DIM,
                                  :]
  BATCH_SIZE = NET.blobs()[0].num  # network batch size
  NUM_OUTPUT = NET.blobs()[-1].channels  # number of output classes
  print("Network batch size is {:d} output size is {:d}".format(BATCH_SIZE, NUM_OUTPUT))

# crop/selective search the images in inputs, extract CNN features, then save to file
def extract_images(inputs):
  # --------------------
  # Assemble into batches
  # --------------------
  # t = time.time()
  image_batches = assemble_batches(inputs, FLAGS.crop_mode, FLAGS.n_max)
  # print('{} batches assembled in {:.3f} s'.format(len(image_batches),
  #                                                 time.time() - t))

  # --------------------
  # Process the batches.
  # --------------------
  t = time.time()
  print 'Processing {} files in {} batches'.format(len(inputs),
                                                   len(image_batches))
  dfs_with_feats = []
  for i in range(len(image_batches)):
    # if i % 10 == 0:
    print('Batch {}/{}, elapsed time: {:.3f} s'.format(i+1,
        len(image_batches),time.time() - t))
    dfs_with_feats.append(compute_feats(image_batches[i]))

  # --------------------
  # Concatenate, droppping the padding rows.
  # --------------------
  df = pd.concat(dfs_with_feats).dropna(subset=['filename'])
  # --------------------
  # Save images
  # --------------------
  t = time.time()
  save_results(df, FLAGS)
  print("Done. Saving to {:d} images took {:.3f} s.".format(len(inputs), time.time() - t))
  
if __name__ == "__main__":
  # Parse cmdline options
  gflags.DEFINE_string(
    "model_def", "", "Model definition file.")
  gflags.DEFINE_string(
    "pretrained_model", "", "Pretrained model weights file.")
  gflags.DEFINE_boolean(
    "gpu", False, "Switch for gpu computation.")
  gflags.DEFINE_string(
    "crop_mode", "center_only", "Crop mode, from {}".format(CROP_MODES))
  gflags.DEFINE_string(
    "input_file", "", "Input txt/csv filename.")
  gflags.DEFINE_string(
    "output_file", "", "Output h5/csv filename.")
  gflags.DEFINE_string(
    "images_dim", 256, "Canonical dimension of (square) images.")
  gflags.DEFINE_string(
    "images_mean_file",
    os.path.join(os.path.dirname(__file__), '../imagenet/ilsvrc_2012_mean.npy'),
    "Data set image mean (numpy array).")
  # AJ:
  gflags.DEFINE_integer(
    "n_max", -1, "maximum number of boxes to mine. -1 means no restriction")
  gflags.DEFINE_boolean(
    "save_each_image", False, "Whether to save features for each image separately or not.")
  gflags.DEFINE_string(
    "out_file_suffix", "", "suffix to follow output file name including the extention (.h5, .mat, .csv).")
  gflags.DEFINE_integer(
    "img_batch_size", 15, "how many images to work on at once")
  gflags.DEFINE_string(
    "image_ext", "", "extention of image (.jpg, .png, etc including the dot) if not supplied")    
  gflags.DEFINE_string("output_dir", "", "output directory")
  gflags.DEFINE_string("image_dir", "", "image directory")

  FLAGS = gflags.FLAGS
  FLAGS(sys.argv)

  if FLAGS.output_file == "":
    FLAGS.save_each_image = True
    # this has to be set to write individually
    assert(FLAGS.out_file_suffix != "")
    
  # --------------------
  # Load input
  # .txt = list of filenames
  # .csv = dataframe that must include a header
  #        with column names filename, ymin, xmin, ymax, xmax
  # --------------------
  print('Loading input and assembling image batches...')
  if FLAGS.input_file.lower().endswith('txt'):
    with open(FLAGS.input_file) as f:
      inputs = [os.path.join(FLAGS.image_dir, _.strip()+FLAGS.image_ext) for _ in f.readlines()]
  elif FLAGS.input_file.lower().endswith('csv'):
    inputs = pd.read_csv(FLAGS.input_file, sep=',', dtype={'filename': str})
    for i in xrange(len(inputs.filename)):
        inputs['filename'][i] = os.path.join(FLAGS.image_dir, inputs.filename[i])

    inputs.set_index('filename', inplace=True)
  else:
    raise Exception("Uknown input file type: not in txt or csv")

  if FLAGS.save_each_image:
    # Check if output file exists or not, if exists, don't process this img
    pattern = '(?<={})(.*)(?=\.(jpg|png|tiff|JPG|JPEG|PNG))'.format(
        FLAGS.image_dir if FLAGS.image_dir[-1]=='/' else FLAGS.image_dir+"/")

    def get_bare_name(pattern, fname):
      match = re.search(pattern, fname)
      return match.group() if match is not None else fname

    out_file = [os.path.join(FLAGS.output_dir,'{}{}'.format(
      get_bare_name(pattern, fname),FLAGS.out_file_suffix)) for fname in inputs]

    new_inputs = [fname for idx, fname in enumerate(inputs) if not os.path.exists(out_file[idx])]

    for fname in new_inputs:
      if not os.path.exists(fname):
        raise Exception("image {} doesn't exist!\n".format(fname))
                        
    print('--- {} out of {} files are already processed, skipping those.. ---'
          .format(len(inputs) - len(new_inputs), len(inputs)))
    inputs = new_inputs

  if len(inputs) == 0:
     sys.exit()
  # --------------------
  # Configure network, input, output
  # --------------------
  config(FLAGS.model_def, FLAGS.pretrained_model, FLAGS.gpu, FLAGS.images_dim,
         FLAGS.images_mean_file)
                      
  if FLAGS.crop_mode == 'selective_search':
    # Work in image batch if running selecitve_search
    nImgs = len(inputs)
    itr = nImgs / FLAGS.img_batch_size if nImgs > FLAGS.img_batch_size else 1
    for i in xrange(itr):
      t = time.time()
      extract_images(inputs[i*FLAGS.img_batch_size:FLAGS.img_batch_size*(i+1)])
      print('--- Image Batch {}/{}, elapsed time: {:.3f} s ---'.format(i+1,itr, \
                                                                       time.time() - t))
  else:
    print('--- Extracting feats with {} for {:d} images---'.format(FLAGS.crop_mode, len(inputs)))
    t = time.time()
    extract_images(inputs)
    print('--- Done extracting feats with {} time: {:.3f} s ---'.format(FLAGS.crop_mode, \
                                                                        time.time() - t))
  
  sys.exit()                                                                   
