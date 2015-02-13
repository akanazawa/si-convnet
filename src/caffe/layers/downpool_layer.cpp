// Written by Angjoo Kanazawa & Abhishek Sharma 2013
#include <vector>
#include <string>
#include <algorithm>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/transformation.hpp"
#include "caffe/util/imshow.hpp"

#include <opencv2/highgui/highgui.hpp>

namespace caffe {

template <typename Dtype>
void DownPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                         vector<Blob<Dtype> *> *top) {
  CHECK_EQ(top->size(), 1)
      << "DownPooling Layer takes a single blob as output.";
  this->NUM_T_ = this->layer_param_.transformations_size();
  CHECK_EQ(bottom.size(), this->NUM_T_) << "DownPooling Layer's input must be "
                                           "the same size as the # of "
                                           "transformations.";

  this->coord_idx_.resize(this->NUM_T_);

  // Assumes that the first bottom is always identity == canonical shape
  NUM_OUTPUT_ = bottom[0]->num();          // canonical num
  CHANNEL_OUTPUT_ = bottom[0]->channels(); // canonical channel
  HEIGHT_ = bottom[0]->height();           // canonical height
  WIDTH_ = bottom[0]->width();             // canonical width
  BORDER_ = static_cast<Border>(this->layer_param_.transformations(0).border());
  INTERP_ = static_cast<Interp>(this->layer_param_.transformations(0).interp());

  // Reshape switch_idx.
  switch_idx_.Reshape(NUM_OUTPUT_, CHANNEL_OUTPUT_, HEIGHT_, WIDTH_);
  // Resizing NUM_T_ here, but we only use 1:NUM_T_-1 top_buffer because the
  // first one is always the identity.
  top_buffer_.resize(NUM_T_);
  int height_new = 0, width_new = 0;
  // Empty placeholder for 0th scale
  coord_idx_[0].reset(new Blob<float>(1, 1, 1, 1, false));

  // Blob that will hold the coordinates of inverse transformation (tmp
  // placeholder buffer before cropping)
  Blob<float> *inverse_coord = new Blob<float>(false);

  for (int i = 1; i < this->NUM_T_; ++i) {
    // Compute Tmat that gives inverse transformations to undo the
    // transformation @ up stage
    TMatFromProto(this->layer_param_.transformations(i), tmat_, true);

    // Generate the coordinate matrix for the reverse transformation.
    GenCoordMat(tmat_, bottom[i]->height(), bottom[i]->width(), inverse_coord,
                height_new, width_new, BORDER_, INTERP_);

    // Transforms into canonical shape
    switch (INTERP_) {
    case NN:
      coord_idx_[i].reset(new Blob<float>(1, 1, HEIGHT_ * WIDTH_, 1, false));
      break;
    case BILINEAR:
      coord_idx_[i].reset(
          new Blob<float>(1, 1, HEIGHT_ * WIDTH_ * 4, 1, false));
      break;
    default:
      LOG(FATAL) << "Unknown pooling method.";
      break;
    }

    // Crop the coordinates tso that output size is always the canonical shape.
    ImageSize original(width_new, height_new);
    ImageSize target(WIDTH_, HEIGHT_);
    CropCenter(inverse_coord->cpu_data(), original, target, INTERP_,
               coord_idx_[i]->mutable_cpu_data());

    top_buffer_[i].reset(
        new Blob<Dtype>(NUM_OUTPUT_, CHANNEL_OUTPUT_, HEIGHT_, WIDTH_, false));
  }

  delete inverse_coord;

  (*top)[0]->Reshape(NUM_OUTPUT_, CHANNEL_OUTPUT_, HEIGHT_, WIDTH_);

  // NUM_T_ + 1 bc the 0th dim is used to count how many times fwd was run so
  // that we can take the average in the end.
  trans_counter_.resize(NUM_T_ + 1);

  CHECK_EQ(1, (*top).size());
};

// Take multiple bottoms turn them into 1 top
template <typename Dtype>
void DownPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                          vector<Blob<Dtype> *> *top) {

  Dtype *top_data = (*top)[0]->mutable_cpu_data();
  int top_count = (*top)[0]->count();
  int sheet_count = (*top)[0]->height() * (*top)[0]->width();
  // Initialize top_data with the first bottom data since this is always
  // identity transform.
  caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(), top_data);
  // montage(bottom[0], "cpu identity");

  const Dtype *buffer_top_data;
  // Get the mutable switch data
  float *switch_data = switch_idx_.mutable_cpu_data();
  std::fill(switch_data, switch_data + switch_idx_.count(), 0);
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX: // MAX
    const Dtype *coord_data;
    for (int i = 1; i < NUM_T_; ++i) {
      // Apply interpolation on bottom_data
      coord_data = this->coord_idx_[i]->cpu_data();
      // montage(bottom[i], this->layer_param_.transformations(i), "cpu b4");
      InterpImageNN_cpu(bottom[i], coord_data, this->top_buffer_[i].get(),
                        INTERP_);
      buffer_top_data = this->top_buffer_[i]->cpu_data();
      // montage(this->top_buffer_[i].get(),
      // this->layer_param_.transformations(i),
      //         "cpu after");
      // Do max and record the max switches.
      for (int countI = 0; countI < top_count; ++countI) {
        // Only update the switch if the value is valid (not 0 padded region);
        if (coord_data[countI % sheet_count] >= 0 &&
            buffer_top_data[countI] > top_data[countI]) {
          switch_data[countI] = i;
          top_data[countI] = buffer_top_data[countI];
        }
      }
    }
    // montage((*top)[0], "cpu max");
    // montage(&switch_idx_, "cpu switch");
    // cv::waitKey(0);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
    break;
  }
  // Count usage.
  int counter_data[NUM_T_];
  for (int t = 0; t < NUM_T_; ++t) {
    counter_data[t] = std::count(switch_data, switch_data + top_count, t);
  }
  // Save the % usage info.
  UpdateCounter(counter_data, top_count);
}

template <typename Dtype>
void DownPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                          vector<Blob<Dtype> *> *top) {
  Dtype *top_data = (*top)[0]->mutable_gpu_data();
  int top_count = (*top)[0]->count();
  int sheet_count = (*top)[0]->height() * (*top)[0]->width();
  // Initialize top_data with the first bottom data since this is always
  // the identity transform.
  caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(), top_data);
  // montage_channels(bottom[0], "identity");
  const Dtype *buffer_top_data; 
  // Get the mutable switch data
  float *switch_data = switch_idx_.mutable_gpu_data();
  CUDA_CHECK(cudaMemset(switch_data, 0, sizeof(Dtype) * switch_idx_.count()));

  switch (this->layer_param_.pooling_param().pool()) {

  case PoolingParameter_PoolMethod_MAX: // MAX
    for (int i = 1; i < this->NUM_T_; ++i) {
      const Dtype *coord_data = coord_idx_[i]->gpu_data();
      // Apply interpolation on bottom_data
      // montage_channels(bottom[i], this->layer_param_.transformations(i),
      // "b4");
      InterpImageNN_gpu(bottom[i], coord_data, top_buffer_[i].get(), INTERP_);
      // montage_channels(top_buffer_[i].get(),
      // this->layer_param_.transformations(i), "after");
      buffer_top_data = top_buffer_[i]->gpu_data();
      // This kernel takes max and saves the max switches.
      MaxTransSetSwitch_gpu(buffer_top_data, top_data, top_count, coord_data,
                            sheet_count, switch_data, i);
    }
    // montage((*top)[0], "max");
    // montage(&switch_idx_, "switch");

    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
    break;
  }

  // Count usage
  // This will probably slow down GPU bc memory switch data has to be copied
  // over to cpu..
  int counter_data[NUM_T_];
  CountSwitches(switch_data, switch_idx_.count(), NUM_T_, counter_data);
  UpdateCounter(counter_data, top_count);

  // Report("");
  // cv::waitKey(0);
}

// One top, multiple bottoms
template <typename Dtype>
void DownPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                           const vector<bool> &propagate_down,
                                           vector<Blob<Dtype> *> *bottom) {

  int tIndex, origRow, origCol, backOffset, bottomOffset;

  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX) {
    // Setting the diff value to all the bottoms to all 0, so that I can
    // incrementally add diff from top to the bottom
    Dtype **bottomMutableDiffPtrs = new Dtype *[this->NUM_T_];
    const Dtype **coord_data = new const Dtype *[this->NUM_T_];
    for (int t = 0; t < NUM_T_; ++t) {
      coord_data[t] = this->coord_idx_[t]->cpu_data();
      bottomMutableDiffPtrs[t] = (*bottom)[t]->mutable_cpu_diff();
      memset((*bottom)[t]->mutable_cpu_diff(), 0,
             sizeof(Dtype) * (*bottom)[t]->count());
    }
    switch (INTERP_) {
    case NN:
      for (int n = 0; n < this->NUM_OUTPUT_; ++n) {
        for (int c = 0; c < this->CHANNEL_OUTPUT_; c++) {
          for (int h = 0; h < this->HEIGHT_; h++) {
            for (int w = 0; w < this->WIDTH_; w++) {
              tIndex =
                  this->switch_idx_.data_at(n, c, h, w); // The transformation
                                                         // index which resulted
                                                         // in the maxed output.
              if (tIndex == 0) {
                (*bottom)[0]->add_diff_at(top[0]->diff_at(n, c, h, w), n, c, h,
                                          w);
              } else {
                // AJ: top diff at n,c,h,w goes to n,c,h,w of
                // bottom_diff[tIndex]
                // now that bottom_diff[tIndex] @ n,c,h,w has to be transformed
                // back to it's original shape
                backOffset = static_cast<int>(
                    (coord_data[tIndex])[h * this->WIDTH_ + w]);
                if (backOffset >= 0) {
                  origRow = backOffset / (*bottom)[tIndex]->width();
                  origCol = backOffset % (*bottom)[tIndex]->width();
                  bottomOffset =
                      (*bottom)[tIndex]->offset(n, c, origRow, origCol);
                  (bottomMutableDiffPtrs[tIndex])[bottomOffset] +=
                      top[0]->diff_at(n, c, h, w);
                }
              }
            }
          }
        }
      } // end of for (int n = 0;..
      break;
    case BILINEAR: {
      int ind_top, r0, c0, ind_p11, r1, c1, offset;
      float dc, dr, w00, w01, w10, w11, top_error;
      int N = this->WIDTH_ * this->HEIGHT_;

      for (int n = 0; n < this->NUM_OUTPUT_; ++n) {
        for (int c = 0; c < this->CHANNEL_OUTPUT_; c++) {
          for (int h = 0; h < this->HEIGHT_; h++) {
            for (int w = 0; w < this->WIDTH_; w++) {
              tIndex =
                  this->switch_idx_.data_at(n, c, h, w); // The transformation
                                                         // index which resulted
                                                         // in the maxed output.
              if (tIndex == 0) {
                (*bottom)[0]->add_diff_at(top[0]->diff_at(n, c, h, w), n, c, h,
                                          w);
              } else {
                ind_top = h * this->WIDTH_ + w;
                backOffset = static_cast<int>((coord_data[tIndex])[ind_top]);
                if (backOffset >= 0) {
                  int width_bottom = (*bottom)[tIndex]->width();
                  int height_bottom = (*bottom)[tIndex]->width();
                  r0 = backOffset / width_bottom;
                  c0 = backOffset % width_bottom;
                  ind_p11 = static_cast<int>((coord_data[tIndex])[ind_top + N]);
                  r1 = ind_p11 / width_bottom;
                  c1 = ind_p11 % width_bottom;

                  dc = (coord_data[tIndex])[ind_top + 2 * N];
                  dr = (coord_data[tIndex])[ind_top + 3 * N];

                  w00 = (1 - dc) * (1 - dr);
                  w01 = (1 - dr) * dc;
                  w10 = (1 - dc) * dr;
                  w11 = dr * dc;

                  offset = (n * this->CHANNEL_OUTPUT_ + c) * height_bottom;
                  top_error = top[0]->diff_at(n, c, h, w);

                  (bottomMutableDiffPtrs[tIndex])[(offset + r0) * width_bottom +
                                                  c0] += w00 * top_error;
                  (bottomMutableDiffPtrs[tIndex])[(offset + r0) * width_bottom +
                                                  c1] += w01 * top_error;
                  (bottomMutableDiffPtrs[tIndex])[(offset + r1) * width_bottom +
                                                  c0] += w10 * top_error;
                  (bottomMutableDiffPtrs[tIndex])[(offset + r1) * width_bottom +
                                                  c1] += w11 * top_error;
                }
              }
            }
          }
        }
      } // end of for (int n = 0;..
    }   // scoping bilinear
    break;
    default:
      LOG(ERROR) << "Unknown interpolation mode " << this->INTERP_;
    } // end of switch
    delete[] bottomMutableDiffPtrs;

  }
}

template <typename Dtype>
void DownPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                           const vector<bool> &propagate_down,
                                           vector<Blob<Dtype> *> *bottom) {
  const Dtype *top_diff = top[0]->gpu_diff();
  const int topCount = top[0]->count();
  const float *switchData = this->switch_idx_.gpu_data();

  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX) {
    // AJ: doing backprop on each input (simpler way that works
    // now):
    for (int t = 0; t < NUM_T_; ++t) {
      // Reset all diffs to 0.
      CUDA_CHECK(cudaMemset((*bottom)[t]->mutable_gpu_diff(), 0,
                            sizeof(Dtype) * (*bottom)[t]->count()));

      ErrorPropagateDownpoolNN_gpu_single(
          top_diff, t, topCount, top[0]->height() * top[0]->width(), switchData,
          this->coord_idx_[t]->gpu_data(), (*bottom)[t]->mutable_gpu_diff(),
          (*bottom)[t]->height() * (*bottom)[t]->width(), (*bottom)[t]->width(),
          this->INTERP_);
    }
  }
}

template <typename Dtype>
void DownPoolingLayer<Dtype>::UpdateCounter(const int *curr_counter,
                                            const int &top_count) {
  // Increment the # of times this was run
  ++trans_counter_[0];
  float usage;
  int total_sum = 0;
  for (int i = 0; i < NUM_T_; ++i) {
    usage = static_cast<float>(curr_counter[i]) / top_count;
    total_sum += curr_counter[i];
    trans_counter_[i + 1] = usage;
  }
  CHECK(total_sum == top_count);
}

template <typename Dtype>
void DownPoolingLayer<Dtype>::Report(const std::string &name) {
  std::string str = "" + name + ", transformation usage: ";
  char buff[100];
  float usage;
  for (int i = 1; i < NUM_T_ + 1; ++i) {
    usage = trans_counter_[i];
    sprintf(buff, "T%d: %.2f%% ", i - 1, usage * 100);
    str += buff;
  }
  LOG(INFO) << str;
  // Reset the counter.
  std::fill(trans_counter_.begin(), trans_counter_.end(), 0);
}

INSTANTIATE_CLASS(DownPoolingLayer);

} // namespace caffe
