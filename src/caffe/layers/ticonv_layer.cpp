// Angjoo Kanazawa
// Implementation of Transformation Invariant Convolution Layer
// Just a wrapper that wraps UpsamplingLayer, TiedConvLayer, DownPoolingLayer
// together.
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/imshow.hpp"

#include <opencv2/highgui/highgui.hpp>

namespace caffe {

template <typename Dtype>
void TIConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                      vector<Blob<Dtype> *> *top) {
  CHECK_EQ(bottom.size(), 1) << "TIConv Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "TIConv Layer takes a single blob as output.";

  // NUM_T_ includes includes Identity transform
  this->NUM_T_ = this->layer_param_.transformations_size();

  LOG(INFO) << "TIConvolution layer using " << NUM_T_ << " transformations "
            << this->layer_param_.name() << " using interpolation: "
            << this->layer_param_.transformations(0).interp();
  TransParameter tparam;
  for (int t = 0; t < this->NUM_T_; ++t) {
    tparam = this->layer_param_.transformations(t);
    LOG(INFO) << " T" << t << " : "
              << " sc: " << tparam.scale() << ", rot: " << tparam.rotation();
  }
  if (Caffe::phase() == Caffe::TRAIN)
    LOG(INFO) << "  Creating Upsampling Layer in " << this->layer_param_.name();
  this->up_layer_ = new UpsamplingLayer<Dtype>(this->layer_param_);
  // The bottom that net provides is the bottom of UP layer
  // for T transformations, need to make blob for each
  for (int t = 0; t < this->NUM_T_; ++t) {
    shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
    activations_.push_back(blob_pointer);
    // and add them to up_top_vec_
    up_top_vec_.push_back(blob_pointer.get());
  }
  // Setup UP layer, print shape
  this->up_layer_->SetUp(bottom, &up_top_vec_);
  if (Caffe::phase() == Caffe::TRAIN) {
    for (int i = 0; i < up_top_vec_.size(); ++i) {
      LOG(INFO) << "  Top shape: " << up_top_vec_[i]->channels() << " "
                << up_top_vec_[i]->height() << " " << up_top_vec_[i]->width();
    }
  }

  // Tied Conv
  if (Caffe::phase() == Caffe::TRAIN)
    LOG(INFO) << "  Creating TiedConv Layer in " << this->layer_param_.name();
  this->tiedconv_layer_ = new TiedConvolutionLayer<Dtype>(this->layer_param_);
  // make new top blobs and add them to tiedconv_top_vec_
  for (int t = 0; t < this->NUM_T_; ++t) {
    shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
    activations_.push_back(blob_pointer);
    tiedconv_top_vec_.push_back(blob_pointer.get());
  }
  // Setup
  this->tiedconv_layer_->SetUp(up_top_vec_, &tiedconv_top_vec_);
  if (Caffe::phase() == Caffe::TRAIN) {
    for (int i = 0; i < up_top_vec_.size(); ++i) {
      LOG(INFO) << "  Top shape: " << tiedconv_top_vec_[i]->channels() << " "
                << tiedconv_top_vec_[i]->height() << " "
                << tiedconv_top_vec_[i]->width();
    }
  }

  // Connect the W and b of tiedconv_layer to TI's blobs_ to make everything
  // seem normal:
  this->blobs_ = tiedconv_layer_->blobs_;

  // DownPool
  if (Caffe::phase() == Caffe::TRAIN)
    LOG(INFO) << "  Creating DownPooling Layer in "
              << this->layer_param_.name();
  this->downpool_layer_ = new DownPoolingLayer<Dtype>(this->layer_param_);
  // no need to make new blobs here bc downpool_bottom_vec_ == tiedconv_top_vec_
  // and the top that net provides is the top for Downpool
  this->downpool_layer_->SetUp(tiedconv_top_vec_, top);
}

template <typename Dtype>
void TIConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                            vector<Blob<Dtype> *> *top) {
  // Up first:
  up_layer_->Forward_cpu(bottom, &up_top_vec_);
  // montage(up_top_vec_[1], this->layer_param_.transformations(1));
  // cv::waitKey(0);
  // Tied
  tiedconv_layer_->Forward_cpu(up_top_vec_, &tiedconv_top_vec_);
  // Down
  downpool_layer_->Forward_cpu(tiedconv_top_vec_, top);
}

template <typename Dtype>
void TIConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                            vector<Blob<Dtype> *> *top) {
  // Up
  up_layer_->Forward_gpu(bottom, &up_top_vec_);
  // Tied
  tiedconv_layer_->Forward_gpu(up_top_vec_, &tiedconv_top_vec_);
  // Down
  downpool_layer_->Forward_gpu(tiedconv_top_vec_, top);
}

template <typename Dtype>
void TIConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                              const vector<bool>& propagate_down,
                                              vector<Blob<Dtype> *> *bottom) {
  // Down first
  downpool_layer_->Backward_cpu(top, propagate_down, &tiedconv_top_vec_);
  // Tied
  tiedconv_layer_->Backward_cpu(tiedconv_top_vec_, propagate_down,
                                        &up_top_vec_);
  // finally Up
  up_layer_->Backward_cpu(up_top_vec_, propagate_down, bottom);

}

template <typename Dtype>
void TIConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                              const vector<bool>& propagate_down,
                                              vector<Blob<Dtype> *> *bottom) {
  // Down first
  downpool_layer_->Backward_gpu(top, propagate_down, &tiedconv_top_vec_);
  // Tied
  tiedconv_layer_->Backward_gpu(tiedconv_top_vec_, propagate_down,
                                        &up_top_vec_);
  // finally Up
  up_layer_->Backward_gpu(up_top_vec_, propagate_down, bottom);
}

template <typename Dtype>
void TIConvolutionLayer<Dtype>::Report(const std::string &name) {
  up_layer_->Report(name + " Up");
  tiedconv_layer_->Report(name + " TiedConv");
  downpool_layer_->Report(name + " DownPool");
}

INSTANTIATE_CLASS(TIConvolutionLayer);
} // namespace caffe
