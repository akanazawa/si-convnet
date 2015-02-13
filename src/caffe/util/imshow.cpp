// Copyright 2014 Angjoo Kanazawa

#include <algorithm>
#include <cstdio>

#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/util/imshow.hpp"

namespace caffe {

// utilities
void display_image(const std::string &window_name, const cv::Mat *img) {
  cv::namedWindow(window_name,
                  CV_WINDOW_NORMAL |
                      CV_WINDOW_KEEPRATIO); // Create a window for display.
  cv::imshow(window_name, *img);
  // change to cv::waitKey(0); to halt for user click.
  cv::waitKey(1);
}

template <typename Dtype>
void imshow(const Blob<Dtype> *image, int show_num, const std::string &prefix,
            bool show_diff) {
  const int channels = image->channels();
  const int height = image->height();
  const int width = image->width();
  const Dtype *data_ptr = show_diff ? image->cpu_diff() : image->cpu_data();
  int show_channels = channels == 3 ? channels : 1;

  int imsize = channels == 3 ? height * width * 3 : height * width;
  int type = channels == 3 ? CV_32FC3 : CV_32FC1;
  Dtype buffer[imsize];
  char name_buff[100];
  std::string window_name;
  for (int i = 0; i < show_num; ++i) {
    // std::copy(data_ptr + image->offset(i, 0, 0, 0),
    //           data_ptr + image->offset(i, 0, 0, 0) + imsize, buffer);
    // open cv does channels for each row
    int counter = 0;
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        for (int c = 0; c < show_channels; ++c) {
          buffer[counter++] = data_ptr[image->offset(i, c, h, w)];
        }
      }
    }

    // between [0, 1]
    normalize(buffer, imsize);

    cv::Mat im(height, width, type, buffer, 0);

    snprintf(name_buff, 100, " %d", i);
    display_image(prefix + name_buff, &im);
  }
}

// Explicit instantiation
template void imshow<float>(const Blob<float> *image, int show_num,
                            const std::string &prefix, bool show_diff);

// Overloading
template <typename Dtype>
void imshow(const Blob<Dtype> *image, const TransParameter &param, int show_num,
            const std::string &prefix, bool show_diff) {
  char buff[200];
  snprintf(buff, 200, " sc %.2f, rot %.1f", param.scale(), param.rotation());
  std::string str(buff);

  imshow(image, show_num, prefix + str, show_diff);
}

// Explicit instantiation
template void imshow<float>(const Blob<float> *image,
                            const TransParameter &param, int show_num,
                            const std::string &prefix, bool show_diff);

// real work happens here
template <typename Dtype>
void get_montage_mat(const Blob<Dtype> *image, bool show_diff,
                     cv::Mat *&result) {
  int num = image->num();
  int channels = image->channels();
  int height = image->height();
  int width = image->width();

  // make into square
  int montage_size = static_cast<int>(ceil(sqrt(static_cast<double>(num))));

  const Dtype *data_ptr = show_diff ? image->cpu_diff() : image->cpu_data();

  int show_channels = channels == 3 ? channels : 1;

  int imsize = channels == 3 ? height * width * 3 : height * width;
  int type = channels == 3 ? CV_32FC3 : CV_32FC1;
  Dtype buffer[imsize];

  int gap_size = ceil(static_cast<float>(height) * 0.1);

  result =
      new cv::Mat(height * montage_size + (montage_size + 1) * gap_size,
                  width * montage_size + (montage_size + 1) * gap_size, type);
  // (*result) = cv::Scalar(0.5, 0.5, 0.5);
  (*result) = cv::Scalar(0.3, 0.3, 0.3);
  // (*result) = cv::Scalar(0, 0, 0);
  int m_row = 0, m_col = 0;
  int current_height = 0, current_width = 0;
  for (int i = 0; i < num; ++i) {
    // 0. make one image
    int counter = 0;
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        for (int c = 0; c < show_channels; ++c) {
          buffer[counter++] = data_ptr[image->offset(i, c, h, w)];
        }
      }
    }
    // between [0, 1]
    normalize(buffer, imsize);

    cv::Mat im(height, width, type, buffer, 0);
    // 1. paste this image to it's position in big image
    m_row = i / montage_size;
    m_col = i % montage_size;

    current_height = m_row * height + (m_row + 1) * gap_size;
    current_width = m_col * width + (m_col + 1) * gap_size;

    cv::Mat to(*result, cv::Range(current_height, current_height + height),
               cv::Range(current_width, current_width + width));
    im.copyTo(to);
  }
}

// montage over channels for the first
template <typename Dtype>
void get_montage_channels_mat(const Blob<Dtype> *image, bool show_diff,
                              cv::Mat *&result) {
  const int channels = image->channels();
  const int height = image->height();
  const int width = image->width();

  // make into square
  int montage_size =
      static_cast<int>(ceil(sqrt(static_cast<double>(channels))));
  const Dtype *data_ptr = show_diff ? image->cpu_diff() : image->cpu_data();

  int imsize = height * width;
  int type = CV_32FC1;
  Dtype buffer[imsize];

  int gap_size = ceil(static_cast<float>(height) * 0.1);

  result =
      new cv::Mat(height * montage_size + (montage_size + 1) * gap_size,
                  width * montage_size + (montage_size + 1) * gap_size, type);
  (*result) = cv::Scalar(0.3, 0.3, 0.3);
  int m_row = 0, m_col = 0;
  int current_height = 0, current_width = 0;
  for (int c = 0; c < channels; ++c) {
    // 0. make one image
    int counter = 0;
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        buffer[counter++] = data_ptr[image->offset(0, c, h, w)];
      }
    }
    // between [0, 1]
    normalize(buffer, imsize);

    cv::Mat im(height, width, type, buffer, 0);
    // 1. paste this image to it's position in big image
    m_row = c / montage_size;
    m_col = c % montage_size;

    current_height = m_row * height + (m_row + 1) * gap_size;
    current_width = m_col * width + (m_col + 1) * gap_size;

    cv::Mat to(*result, cv::Range(current_height, current_height + height),
               cv::Range(current_width, current_width + width));
    im.copyTo(to);
  }
}

template <typename Dtype>
void montage(const Blob<Dtype> *image, const std::string &prefix,
             bool show_diff) {

  cv::Mat *result = NULL;
  get_montage_mat(image, show_diff, result);

  display_image(prefix, result);
}

template void montage<float>(const Blob<float> *image,
                             const std::string &prefix, bool show_diff);
// Overloading
template <typename Dtype>
void montage(const Blob<Dtype> *image, const TransParameter &param,
             const std::string &prefix, bool show_diff) {
  char buff[200];
  snprintf(buff, 200, " sc %.2f, rot %.1f", param.scale(), param.rotation());
  std::string str(buff);

  montage(image, prefix + str, show_diff);
}

template void montage<float>(const Blob<float> *image,
                             const TransParameter &param,
                             const std::string &prefix, bool show_diff);

template <typename Dtype>
void montage_channels(const Blob<Dtype> *image, const std::string &prefix,
                      bool show_diff) {

  cv::Mat *result = NULL;
  get_montage_channels_mat(image, show_diff, result);
  display_image(prefix, result);
}

template void montage_channels<float>(const Blob<float> *image,
                                      const std::string &prefix,
                                      bool show_diff);
// Overloading
template <typename Dtype>
void montage_channels(const Blob<Dtype> *image, const TransParameter &param,
                      const std::string &prefix, bool show_diff) {
  char buff[200];
  snprintf(buff, 200, " sc %.2f, rot %.1f", param.scale(), param.rotation());
  std::string str(buff);

  montage_channels(image, prefix + str, show_diff);
}

template void montage_channels<float>(const Blob<float> *image,
                                      const TransParameter &param,
                                      const std::string &prefix,
                                      bool show_diff);

template <typename Dtype>
void save_montage(const Blob<Dtype> *image, const std::string &name,
                  bool show_diff) {

  cv::Mat *result = NULL;
  get_montage_mat(image, show_diff, result);

  // std::string window_name = name;
  // cv::namedWindow(window_name, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO );//
  // Create a window for display.
  // cv::imshow( window_name, *result);
  // cv::waitKey(1);

  const int kMinCellHeight = 40;
  if (image->height() < kMinCellHeight) {
    float factor = kMinCellHeight / image->height();
    // cv::resize(*result, *result, cv::Size(0, 0), factor, factor,
    // cv::INTER_NEAREST);
    cv::resize(*result, *result, cv::Size(0, 0), factor, factor);
  }
  result->convertTo(*result, CV_16UC3, 255);
  vector<int> params;
  params.push_back(CV_IMWRITE_JPEG_QUALITY);
  params.push_back(100);
  cv::imwrite(name + ".jpg", *result, params);
  LOG(INFO) << "saving image to " << name;
}
template void save_montage<float>(const Blob<float> *image,
                                  const std::string &name, bool show_diff);

// Given a (x, y), propagate the receptive field
template <typename Dtype>
void prop_region(Net<Dtype> *net, int layer_id, int &r0, int &c0, int &r1,
                 int &c1) {
  const vector<shared_ptr<Layer<Dtype> > >& layers = net->layers();
  int stride, k, pad;
  for (int l = layer_id; l >= 0; --l) {
    const ConvolutionParameter &param =
        layers[l]->layer_param().convolution_param();
    if (!param.has_stride() && !param.has_kernel_size() && !param.has_pad())
      continue;
    stride = param.stride();
    k = param.has_kernel_size() ? param.kernel_size() : 1;
    pad = param.has_pad() ? param.pad() : 0;
    r0 = stride * r0 - pad;
    c0 = stride * c0 - pad;
    r1 = stride * r1 - pad + k - 1;
    c1 = stride * c1 - pad + k - 1;
    // LOG(INFO) << "at layer " << l <<  ", (" << r0 << ", " << c0 << "), ("
    // 	      << r1 << ", " << c1 << ")";
  }
}

// explicit..
template void prop_region<float>(Net<float> *net, int layer_id, int &r0,
                                 int &c0, int &r1, int &c1);

// normalize data between [0, 1]
template <typename Dtype> void normalize(Dtype *data, int n) {
  Dtype max_val = *(std::max_element(data, data + n));
  Dtype min_val = *(std::min_element(data, data + n));
  // Dtype min_val = 0;
  // for ( int i = 0; i < n ; ++i ) {
  //   if (data[i] < min_val && data[i] != 0 )
  //     min_val = data[i];
  // }
  Dtype diff = max_val - min_val;
  if (diff == 0)
    return;
  for (int i = 0; i < n; ++i) {
    // if ( data[i] == 0 )
    //   data[i] = min_val;
    data[i] = (data[i] - min_val) / diff;
  }
}

// Explicit instantiation
template void normalize<float>(float *data, int n);

} // namespace caffe
