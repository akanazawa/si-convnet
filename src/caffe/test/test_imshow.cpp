// implemented by Angjoo Kanazawa, Abhishek Sharma

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <google/protobuf/text_format.h>

#include "gtest/gtest.h"
#include "caffe/util/imshow.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

TEST(imshow, Cat) {
  // If this test is failing, make sure you run the test executable from the
  // base CAFFE_DIR where Makefile lives.
  cv::Mat img = cv::imread(CMAKE_SOURCE_DIR "caffe/test/test_data/cat.jpg", CV_LOAD_IMAGE_GRAYSCALE);

  img.convertTo(img, CV_32FC1, 1./255);
  cv::namedWindow( "from opencv", CV_WINDOW_NORMAL );
  cv::imshow( "from opencv", img );
  cv::waitKey(1);

  int height = img.rows;
  int width = img.cols;

  // Copy image over to blob.
  Blob<float> blob(2, 1, height, width);
  float* data = blob.mutable_cpu_data();
    
  if ( img.isContinuous() ) {
    width *= height;
    height = 1;
  }

  for ( int i = 0; i < height; ++i ) {
    for ( int j = 0; j < width; ++j ) {
      data[i*width + j] = img.at<float>(i,j);
      data[i*width + j + width*height] = img.at<float>(i,j);
    }
  }
  
  TransParameter param; 
  param.set_scale(0.5f);
  param.set_rotation(15.f);
  imshow(&blob, param, 2, "with param");
}

TEST(imshow, montage) {
  cv::Mat img = cv::imread("/nfshomes/kanazawa/Pictures/cat.jpg", CV_LOAD_IMAGE_COLOR);
  img.convertTo(img, CV_32FC3, 1./255);

  int height = img.rows;
  int width = img.cols;
  int channel = img.channels();

  // copy image over to blob
  Blob<float> blob(4, channel, height, width);
  float* data = blob.mutable_cpu_data();
    
  for ( int i = 0; i < height; ++i ) {
    for ( int j = 0; j < width; ++j ) {
      cv::Vec3f intensity = img.at<cv::Vec3f>(i, j);
      for (int c = 0; c < channel; ++c ) {        
        data[(c*height+i)*width + j] = intensity[c];
        data[(c*height+i)*width + j + width*height*channel] = intensity[(c+1)%channel];
        data[(c*height+i)*width + j + 3*width*height*channel] = 1 - intensity[c];
      }
    }
  }
  montage(&blob, "test");
}

}
