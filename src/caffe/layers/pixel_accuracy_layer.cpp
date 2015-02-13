#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void PixelAccuracyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                           vector<Blob<Dtype> *> *top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();
  ignore_zero_ = this->layer_param_.label_param().ignore_label_zero();
}

template <typename Dtype>
void PixelAccuracyLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                        vector<Blob<Dtype> *> *top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_LE(top_k_, bottom[0]->channels())
      << "top_k must be less than or equal to the number of classes.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), bottom[0]->height());
  CHECK_EQ(bottom[1]->width(), bottom[0]->width());
  (*top)[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void PixelAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                            vector<Blob<Dtype> *> *top) {
  Dtype accuracy = 0;
  const int num = bottom[0]->num();
  const int num_classes = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  vector<Dtype> maxval(top_k_ + 1);
  vector<int> max_id(top_k_ + 1);
  int num_relevant_pixels = 0;
  for (int n = 0; n < num; ++n) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        int pixel_label = static_cast<int>(bottom[1]->data_at(n, 0, h, w));
        if (!ignore_zero_ || pixel_label != 0) {
          ++num_relevant_pixels;
          // Adjust label offset if necessary.
          if (ignore_zero_)
            --pixel_label;
          // Top-k accuracy for each pixel.
          std::vector<std::pair<Dtype, int> > bottom_data_vector;
          for (int c = 0; c < num_classes; ++c) {
            bottom_data_vector.push_back(
                std::make_pair(bottom[0]->data_at(n, c, h, w), c));
          }
          std::partial_sort(
              bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
              bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
          // check if true label is in top k predictions
          for (int k = 0; k < top_k_; k++) {
            if (bottom_data_vector[k].second == pixel_label) {
              ++accuracy;
              break;
            }
          }
        }
      }
    }
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  (*top)[0]->mutable_cpu_data()[0] = accuracy / num_relevant_pixels;
}

INSTANTIATE_CLASS(PixelAccuracyLayer);

} // namespace caffe
