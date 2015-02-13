// Angjoo Kanazawa 2014

#ifndef CAFFE_UTIL_IMSHOW_H_
#define CAFFE_UTIL_IMSHOW_H_

#include "caffe/blob.hpp"
#include "caffe/net.hpp"
#include <opencv2/core/core.hpp>

namespace caffe {

template<typename Dtype>
void imshow(const Blob<Dtype>* image, int show_num = 1, 
            const std::string& prefix = "", bool show_diff = false);
template<typename Dtype>
void imshow(const Blob<Dtype>* image, const TransParameter& param, 
            int show_num = 1, const std::string& prefix = "", bool show_diff = false);

template<typename Dtype>
void get_montage_mat(const Blob<Dtype>* image, bool show_diff, cv::Mat*& result);
template<typename Dtype>
void montage(const Blob<Dtype>* image, const std::string& prefix = "", 
	     bool show_diff = false);
template<typename Dtype>
void montage(const Blob<Dtype>* image, const TransParameter& param, 
	     const std::string& prefix = "", 
	     bool show_diff = false);

template<typename Dtype>
void get_montage_channels_mat(const Blob<Dtype>* image, bool show_diff, cv::Mat*& result);
template<typename Dtype>
void montage_channels(const Blob<Dtype>* image, const std::string& prefix = "", 
	     bool show_diff = false);
template<typename Dtype>
void montage_channels(const Blob<Dtype>* image, const TransParameter& param, 
	     const std::string& prefix = "", 
	     bool show_diff = false);


template<typename Dtype>
void save_montage(const Blob<Dtype>* image, const std::string& fname = "", 
		  bool show_diff = false);
void display_image(const std::string& window_name, const cv::Mat* img);


template<typename Dtype> void normalize(Dtype* data, int n);

template<typename Dtype>
void prop_region(const Net<Dtype>* net, int layer_id, int& r0, int& c0, int& r1, int& c1);


} // namespace caffe

#endif   // CAFFE_UTIL_IMSHOW_H_
