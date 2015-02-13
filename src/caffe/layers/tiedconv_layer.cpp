// Written by Angjoo Kanazawa & Abhishek Sharma 2013
// Implementation of Transformation Invariant Convolution Layer
// The transformation parameter are set in the transformation variable in layer
// parameters.

#include <vector>
#include <boost/lexical_cast.hpp>
#include <cstdio>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/imshow.hpp"

#include <opencv2/highgui/highgui.hpp>
namespace caffe {

template <typename Dtype>
void
TiedConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                        vector<Blob<Dtype> *> *top) {
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  CHECK(!conv_param.has_kernel_size() !=
        !(conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(conv_param.has_kernel_size() ||
        (conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!conv_param.has_pad() && conv_param.has_pad_h() &&
         conv_param.has_pad_w()) ||
        (!conv_param.has_pad_h() && !conv_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!conv_param.has_stride() && conv_param.has_stride_h() &&
         conv_param.has_stride_w()) ||
        (!conv_param.has_stride_h() && !conv_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (conv_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = conv_param.kernel_size();
  } else {
    kernel_h_ = conv_param.kernel_h();
    kernel_w_ = conv_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!conv_param.has_pad_h()) {
    pad_h_ = pad_w_ = conv_param.pad();
  } else {
    pad_h_ = conv_param.pad_h();
    pad_w_ = conv_param.pad_w();
  }
  if (!conv_param.has_stride_h()) {
    stride_h_ = stride_w_ = conv_param.stride();
  } else {
    stride_h_ = conv_param.stride_h();
    stride_w_ = conv_param.stride_w();
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->channels();
  num_output_ = conv_param.num_output();
  CHECK_GT(num_output_, 0);
  group_ = conv_param.group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";

  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  bias_term_ = conv_param.bias_term();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(
        new Blob<Dtype>(num_output_, channels_ / group_, kernel_h_, kernel_w_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(
        GetFiller<Dtype>(conv_param.weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the biases:
    // 1 x 1 x 1 x output channels.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, num_output_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          conv_param.bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
};

template <typename Dtype>
void TiedConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                          vector<Blob<Dtype> *> *top) {
  num_in_ = bottom.size();           // total number of data to convolve
  num_ = bottom[0]->num();           // batch
  CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
    " convolution kernel.";
  height_.resize(num_in_);
  width_.resize(num_in_);
  N_.resize(num_in_);

  for (int i = 0; i < num_in_; ++i) {
    CHECK_EQ(channels_, bottom[i]->channels())
        << "channels has to be the same for all bottoms";
    CHECK_EQ(num_, bottom[i]->num())
        << "batch size has to be the same for all bottoms";
    height_[i] = bottom[i]->height();
    width_[i] = bottom[i]->width();
  }
  // Prepare the matrix multiplication computation.
  M_ = num_output_ / group_;
  K_ = channels_ * kernel_h_ * kernel_w_ / group_;
  col_buffers_.resize(num_in_);
  if (bias_term_) {
    bias_multipliers_.resize(num_in_);
  }
  for (int i = 0; i < num_in_; ++i) {
    // The im2col result buffer would only hold one image at a time to avoid
    // overly large memory usage.
    int height_out = (height_[i] + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
    int width_out = (width_[i] + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
    // AJ: Conserve on memory by not making diff on col_buffer.
    this->col_buffers_[i].reset(new Blob<Dtype>(
        1, channels_ * kernel_h_ * kernel_w_, height_out, width_out, false));
    // Figure out the dimensions for individual gemms.
    N_[i] = height_out * width_out;
    (*top)[i]->Reshape(num_, num_output_, height_out, width_out);

    if (bias_term_) {
      // Set up the all ones "bias multiplier" for adding biases by BLAS
      // bias_multipliers_[i].reset(new Blob<Dtype>(1, 1, 1, N_[i], false));
      // caffe_set(N_[i], Dtype(1), bias_multipliers_[i].mutable_cpu_data());
      bias_multipliers_[i].reset(new SyncedMemory(N_[i] * sizeof(Dtype)));
      Dtype *bias_multiplier_data =
          reinterpret_cast<Dtype *>(bias_multipliers_[i]->mutable_cpu_data());
      for (int j = 0; j < N_[i]; ++j) {
        bias_multiplier_data[j] = 1.;
      }
    }
  }
}

template <typename Dtype>
void
TiedConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                         vector<Blob<Dtype> *> *top) {
  const Dtype *weight = this->blobs_[0]->cpu_data();
  const int weight_offset = M_ * K_; // number of filter parameters in a group
  for (int i = 0; i < num_in_; ++i) {
    //-----Same concept as Forward_cpu of convolutionlayer-----
    const Dtype *bottom_data = bottom[i]->cpu_data();
    const int col_offset = K_ * N_[i];
    const int top_offset = M_ * N_[i];
    Dtype *top_data = (*top)[i]->mutable_cpu_data();
    Dtype *col_data = this->col_buffers_[i]->mutable_cpu_data();
    for (int n = 0; n < num_; ++n) {
      // im2col transformation: unroll input regions for filtering
      // into column matrix for multplication.
      im2col_cpu(bottom_data + bottom[i]->offset(n), channels_, height_[i],
                 width_[i], kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_,
                 stride_w_, col_data);
      // Take innerproduct for groups.
      for (int g = 0; g < group_; ++g) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_[i], K_,
                              (Dtype)1., weight + weight_offset * g,
                              col_data + col_offset * g, (Dtype)0.,
                              top_data + (*top)[i]->offset(n) + top_offset * g);
      }
      // Add bias.
      if (bias_term_) {
        caffe_cpu_gemm<Dtype>(
            CblasNoTrans, CblasNoTrans, num_output_, N_[i], 1, (Dtype)1.,
            this->blobs_[1]->cpu_data(),
            reinterpret_cast<const Dtype *>(bias_multipliers_[i]->cpu_data()),
            (Dtype)1., top_data + (*top)[i]->offset(n));
      }
    }
    //---------------------------------------------------------
  }
}

template <typename Dtype>
void
TiedConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                         vector<Blob<Dtype> *> *top) {

  const Dtype *weight = this->blobs_[0]->gpu_data();
  const int weight_offset = M_ * K_;
  for (int i = 0; i < num_in_; ++i) {
    //-----Same concept as Forward_gpu of convolutionlayer-----
    const Dtype *bottom_data = bottom[i]->gpu_data();
    const int col_offset = K_ * N_[i];
    const int top_offset = M_ * N_[i];
    Dtype *top_data = (*top)[i]->mutable_gpu_data();
    Dtype *col_data = this->col_buffers_[i]->mutable_gpu_data();
    for (int n = 0; n < num_; ++n) {
      // First, im2col
      im2col_gpu(bottom_data + bottom[i]->offset(n), channels_, height_[i],
                 width_[i], kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_,
                 stride_w_, col_data);
      // Second, innerproduct with groups.
      for (int g = 0; g < group_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_[i], K_,
                              (Dtype)1., weight + weight_offset * g,
                              col_data + col_offset * g, (Dtype)0.,
                              top_data + (*top)[i]->offset(n) + top_offset * g);
      }
      // third, add bias
      if (bias_term_) {
        caffe_gpu_gemm<Dtype>(
            CblasNoTrans, CblasNoTrans, num_output_, N_[i], 1, (Dtype)1.,
            this->blobs_[1]->gpu_data(),
            reinterpret_cast<const Dtype *>(bias_multipliers_[i]->gpu_data()),
            (Dtype)1., top_data + (*top)[i]->offset(n));
      }
    }
    //---------------------------------------------------------
  }
  // montage(this->blobs_[0].get(), "tconv" +
  // boost::lexical_cast<std::string>(M_));
}

template <typename Dtype>
void
TiedConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                          const vector<bool> &propagate_down,
                                          vector<Blob<Dtype> *> *bottom) {
  //-----Same concept as Backward_cpu of convolutionlayer-----
  // but multiple times for each bottom-top pair, and accumulating dW
  const Dtype *weight = NULL;
  Dtype *weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->cpu_data();
    weight_diff = this->blobs_[0]->mutable_cpu_diff();
    // Init weight diff to all 0s.
    caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  // bias gradient if necessary
  Dtype *bias_diff = NULL;
  if (bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
    caffe_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }

  const int weight_offset = M_ * K_;
  for (int i = 0; i < num_in_; ++i) {
    const Dtype *top_diff = NULL;
    // Bias gradient if necessary.
    if (bias_term_ && this->param_propagate_down_[1]) {
      top_diff = top[i]->cpu_diff();
      for (int n = 0; n < num_; ++n) {
        caffe_cpu_gemv<Dtype>(
            CblasNoTrans, num_output_, N_[i], 1., top_diff + top[i]->offset(n),
            reinterpret_cast<const Dtype *>(bias_multipliers_[i]->cpu_data()),
            1., bias_diff);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      if (!top_diff) {
        top_diff = top[i]->cpu_diff();
      }
      Dtype* col_data = this->col_buffers_[i]->mutable_cpu_data();
      const Dtype* bottom_data = (*bottom)[i]->cpu_data();
      Dtype* bottom_diff = (*bottom)[i]->mutable_cpu_diff();

      const int col_offset = K_ * N_[i];
      const int top_offset = M_ * N_[i];
      for (int n = 0; n < num_; ++n) {
	// Since we saved memory in the forward pass by not storing all col
	// data, we will need to recompute them.
	im2col_cpu(bottom_data + (*bottom)[i]->offset(n), channels_, height_[i],
		   width_[i], kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_,
		   stride_w_, col_data);
	// gradient w.r.t. weight. Note that we will accumulate diffs.
	// AJ: propagate error Delta W_ij = error from above * this_activation^T
        if (this->param_propagate_down_[0]) {
	  for (int g = 0; g < group_; ++g) {
	    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_[i],
				  (Dtype)1.,
				  top_diff + top[i]->offset(n) + top_offset * g,
				  col_data + col_offset * g, (Dtype)1.,
				  weight_diff + weight_offset * g);
	  }
	}
	// gradient w.r.t. bottom data, if necessary
	// AJ: error here = W*error from above
	if (propagate_down[i]) {
          if (weight == NULL) {
            weight = this->blobs_[0]->cpu_data();
          }
	  for (int g = 0; g < group_; ++g) {
	    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_[i], M_,
				  (Dtype)1., weight + weight_offset * g,
				  top_diff + top[i]->offset(n) + top_offset * g,
				  (Dtype)0., col_data + col_offset * g);
	  }
	  // col2im back to the data
	  col2im_cpu(col_data, channels_, height_[i], width_[i], kernel_h_,
		     kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
		     bottom_diff + (*bottom)[i]->offset(n));
	}
      }
    }
    //---------------------------------------------------------
  }
}

template <typename Dtype>
void
TiedConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                          const vector<bool> &propagate_down,
                                          vector<Blob<Dtype> *> *bottom) {
  const Dtype *weight = NULL;
  Dtype *weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
    // Init weight diffs to all 0s.
    caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  Dtype *bias_diff = NULL;
  if (bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }

  const int weight_offset = M_ * K_;
  for (int i = 0; i < num_in_; ++i) {
    //-----Same concept as Backward_cpu of convolutionlayer-----
    const Dtype* top_diff = NULL;
    // Bias gradient if necessary
    if (bias_term_ && this->param_propagate_down_[1]) {
      top_diff = top[i]->gpu_diff();
      for (int n = 0; n < num_; ++n) {
        caffe_gpu_gemv<Dtype>(
            CblasNoTrans, num_output_, N_[i], 1., top_diff + top[i]->offset(n),
            reinterpret_cast<const Dtype *>(bias_multipliers_[i]->gpu_data()),
            1., bias_diff);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      if (!top_diff) {
        top_diff = top[i]->gpu_diff();
      }
      Dtype* col_data = this->col_buffers_[i]->mutable_gpu_data();
      const Dtype* bottom_data = (*bottom)[i]->gpu_data();
      Dtype* bottom_diff = (*bottom)[i]->mutable_gpu_diff();

      const int col_offset = K_ * N_[i];
      const int top_offset = M_ * N_[i];
      for (int n = 0; n < num_; ++n) {
	// Since we saved memory in the forward pass by not storing all col data,
	// we will need to recompute them.
	im2col_gpu(bottom_data + (*bottom)[i]->offset(n), channels_, height_[i],
                   width_[i], kernel_h_, kernel_w_, pad_h_, pad_w_,
                   stride_h_, stride_w_, col_data);
	// gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
	  for (int g = 0; g < group_; ++g) {
	    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_[i],
				  (Dtype)1.,
				  top_diff + top[i]->offset(n) + top_offset * g,
				  col_data + col_offset * g, (Dtype)1.,
				  weight_diff + weight_offset * g);
	  }
	}
	// gradient w.r.t. bottom data, if necessary
	if (propagate_down[i]) {
          if (weight == NULL) {
            weight = this->blobs_[0]->gpu_data();
          }
	  for (int g = 0; g < group_; ++g) {
	    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_[i], M_,
				  (Dtype)1., weight + weight_offset * g,
				  top_diff + top[i]->offset(n) + top_offset * g,
				  (Dtype)0., col_data + col_offset * g);
	  }
	  // col2im back to the data
	  col2im_gpu(col_data, channels_, height_[i], width_[i], kernel_h_,
		     kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
		     bottom_diff + (*bottom)[i]->offset(n));
	}
      }
    }
    // montage_channels(this->blobs_[0].get(),
    // boost::lexical_cast<std::string>(M_) + " tconv bprop " +
    // boost::lexical_cast<std::string>(i) , true);
    //// make sure to give back the pointer to gpu after visualization
    // weight_diff = this->blobs_[0]->mutable_gpu_diff();
  } // end for each input
    // montage_channels(this->blobs_[0].get(), "final tconv bprop " +
    // boost::lexical_cast<std::string>(M_), true);
    // cv::waitKey(0);
}

template <typename Dtype>
void TiedConvolutionLayer<Dtype>::Report(const std::string &name) {
  // montage(this->blobs_[0].get(), name + " tconv");
  // montage(this->blobs_[0].get(), name + " tconv bprop", true);
  // // cv::waitKey(0);
  // fprintf(stderr, "%s tied-conv W_diff:\n", name.c_str());
  // Blob<Dtype>* W = this->blobs_[0].get();
  // for (int h = 0; h < W->height(); ++h) {
  //   for (int w = 0; w < W->width(); ++w) {
  //     fprintf(stderr, "%.2g ", (W->diff_at(0, 0, h, w)));
  //   }
  //   fprintf(stderr, "\n");
  // }
}

INSTANTIATE_CLASS(TiedConvolutionLayer);
} // namespace caffe
