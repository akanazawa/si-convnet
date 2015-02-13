// Written by Angjoo Kanazawa & Abhishek Sharma 2013
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/transformation.hpp"
#include "caffe/util/imshow.hpp"

namespace caffe {

template <typename Dtype>
void UpsamplingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                        vector<Blob<Dtype> *> *top) {
  CHECK_EQ(bottom.size(), 1)
      << "Upsampling Layer takes a single blob as input.";

  NUM_T_ = this->layer_param_.transformations_size();
  CHECK_GT(NUM_T_, 0) << "NUM_T_ must be at least 1 (identity)";
  CHECK_EQ(top->size(), NUM_T_) << "Upsampling Layer's output has to be the "
                                   "same size as the # of transformations.";
  coord_idx_.resize(NUM_T_);

  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  HEIGHT_ = bottom[0]->height();
  WIDTH_ = bottom[0]->width();
  BORDER_ = static_cast<Border>(this->layer_param_.transformations(0).border());
  INTERP_ = static_cast<Interp>(this->layer_param_.transformations(0).interp());

  int height_new = 0, width_new = 0;
  for (int i = 0; i < NUM_T_; ++i) {
    if (this->layer_param_.transformations(i).scale() == 1 &&
        this->layer_param_.transformations(i).rotation() == 0) {
      // If transformation is identity, don't do anything.
      (*top)[i]->Reshape(num, channels, HEIGHT_, WIDTH_);
    } else {
      // Compute transformation.
      TMatFromProto(this->layer_param_.transformations(i), tmat_);
      if (!this->layer_param_.transformations(i).has_final_width()) {
        // Initialize coord_idx ptr, don't make diff on this blob to save memory.
        coord_idx_[i].reset(new Blob<float>(false)); 
        GenCoordMat(tmat_, HEIGHT_, WIDTH_, coord_idx_[i].get(), height_new,
                    width_new, BORDER_, INTERP_);
      } else {
        // Canoincal size is set, so after finding the transformation,
        // crop or pad to that canonical size.
        // First find the coordinate matrix for this transformation
        Blob<float> *original_coord = new Blob<float>(false);
        GenCoordMat(tmat_, HEIGHT_, WIDTH_, original_coord, height_new,
                    width_new, BORDER_, INTERP_);
        LOG(INFO) << "Cropping " << height_new << " by " << width_new
                  << " to specified size: "
                  << this->layer_param_.transformations(i).final_height()
                  << " by "
                  << this->layer_param_.transformations(i).final_width();
        // Crop the coordinates at the center to this size.
        const int orig_width = width_new;
        const int orig_height = height_new;
        width_new = this->layer_param_.transformations(i).final_width();
        height_new = this->layer_param_.transformations(i).final_height();

        // Need to set coordinates to the size of crop_size.
        switch (INTERP_) {
        case NN:
          coord_idx_[i].reset(
              new Blob<float>(1, 1, height_new * width_new, 1, false));
          break;
        case BILINEAR:
          coord_idx_[i].reset(
              new Blob<float>(1, 1, height_new * width_new * 4, 1, false));
          break;
        default:
          LOG(FATAL) << "Unknown pooling method.";
          break;
        }
        ImageSize original(orig_width, orig_height);
        ImageSize target(width_new, height_new);
        CropCenter(original_coord->cpu_data(), original, target, INTERP_,
                   coord_idx_[i]->mutable_cpu_data());

        // Clean up.
        delete original_coord;
      }
      // need to set the size of the corresponding top
      (*top)[i]->Reshape(num, channels, height_new, width_new);
    }
  }

  CHECK_EQ(NUM_T_, (*top).size());
};

template <typename Dtype>
void UpsamplingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                         vector<Blob<Dtype> *> *top) {

  for (int i = 0; i < NUM_T_; ++i) {
    if (this->layer_param_.transformations(i).scale() == 1 &&
        this->layer_param_.transformations(i).rotation() == 0) {
      // Simply copy for identity transformation.
      caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(),
                 (*top)[i]->mutable_cpu_data());
    } else {
      // apply Interpolation on bottom_data using tmat_[i] into top_data
      // the coord_idx_[i] will be of size newH_ \times newW_, which is pre-set
      InterpImageNN_cpu(bottom[0], coord_idx_[i]->cpu_data(), (*top)[i],
                        INTERP_);
    }
  }
}

template <typename Dtype>
void UpsamplingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                         vector<Blob<Dtype> *> *top) {
  for (int i = 0; i < NUM_T_; ++i) {
    if (this->layer_param_.transformations(i).scale() == 1 &&
        this->layer_param_.transformations(i).rotation() == 0) {
      // Simply copy over for identity transformation.
      caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(),
                 (*top)[i]->mutable_gpu_data());
      // montage_channels((*top)[0], "up: identity");
      // imshow((*top)[0], 1, "up: identity");
    } else {
      // Apply Interpolation on bottom_data using tmat_[i] into top_data.
      InterpImageNN_gpu(bottom[0], coord_idx_[i]->gpu_data(), (*top)[i],
                        INTERP_);
    }
    // montage((*top)[i], this->layer_param_.transformations(i), "up:");
  }
}

// Backward has to return 1 bottom
// Note that backwards coordinate indices are also stored in data
template <typename Dtype>
void UpsamplingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                          const vector<bool> &propagate_down,
                                          vector<Blob<Dtype> *> *bottom) {
  // Reset bottom diff.
  caffe_set((*bottom)[0]->count(), Dtype(0.), (*bottom)[0]->mutable_cpu_diff());
  if (propagate_down[0]) {
    for (int i = 0; i < NUM_T_; ++i) {
      if (this->layer_param_.transformations(i).scale() == 1 &&
          this->layer_param_.transformations(i).rotation() == 0) {
        // Simply copy for identity transformation.
        caffe_copy(top[i]->count(), top[i]->cpu_diff(),
                   (*bottom)[i]->mutable_cpu_diff());
      } else {
        PropagateErrorNN_cpu(top[i], coord_idx_[i]->cpu_data(), (*bottom)[0],
                             INTERP_);
        // Assumes that only one bottom is supplied as input.
      }
    }
  }
}

template <typename Dtype>
void UpsamplingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                          const vector<bool> &propagate_down,
                                          vector<Blob<Dtype> *> *bottom) {
  // Reset bottom diff.
  caffe_gpu_set((*bottom)[0]->count(), Dtype(0.),
                (*bottom)[0]->mutable_gpu_diff());

  if (propagate_down[0]) {
    for (int i = 0; i < NUM_T_; ++i) {
      if (this->layer_param_.transformations(i).scale() == 1 &&
          this->layer_param_.transformations(i).rotation() == 0) {
        caffe_copy(top[i]->count(), top[i]->gpu_diff(),
                   (*bottom)[i]->mutable_gpu_diff());
      } else {
        PropagateErrorNN_gpu(top[i], coord_idx_[i]->gpu_data(), (*bottom)[0],
                             INTERP_);
      }
      // montage((*bottom)[0], this->layer_param_.transformations(i),
      // "up-back:");
    }
  }
}

INSTANTIATE_CLASS(UpsamplingLayer);

} // namespace caffe
