#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class PixelAccuracyLayerTest : public ::testing::Test {
protected:
  PixelAccuracyLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(100, 10, 5, 3)),
        blob_bottom_label_(new Blob<Dtype>(100, 1, 5, 3)),
        blob_top_(new Blob<Dtype>()), top_k_(3) {
    // fill the probability values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);

    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    shared_ptr<Caffe::RNG> rng(new Caffe::RNG(prefetch_rng_seed));
    caffe::rng_t *prefetch_rng = static_cast<caffe::rng_t *>(rng->generator());
    Dtype *label_data = blob_bottom_label_->mutable_cpu_data();
    for (int i = 0; i < 100 * 5 * 3; ++i) {
      label_data[i] = (*prefetch_rng)() % 10;
    }
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~PixelAccuracyLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_;
  }
  Blob<Dtype> *const blob_bottom_data_;
  Blob<Dtype> *const blob_bottom_label_;
  Blob<Dtype> *const blob_top_;
  vector<Blob<Dtype> *> blob_bottom_vec_;
  vector<Blob<Dtype> *> blob_top_vec_;
  int top_k_;
};

TYPED_TEST_CASE(PixelAccuracyLayerTest, TestDtypes);

TYPED_TEST(PixelAccuracyLayerTest, TestSetup) {
  LayerParameter layer_param;
  PixelAccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(PixelAccuracyLayerTest, TestSetupTopK) {
  LayerParameter layer_param;
  AccuracyParameter *accuracy_param = layer_param.mutable_accuracy_param();
  accuracy_param->set_top_k(5);
  PixelAccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(PixelAccuracyLayerTest, TestForwardCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  PixelAccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));

  const int num = this->blob_bottom_data_->num();
  const int num_classes = this->blob_bottom_data_->channels();
  const int height = this->blob_bottom_data_->height();
  const int width = this->blob_bottom_data_->width();
  int num_correct_labels = 0;
  for (int i = 0; i < num; ++i) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        TypeParam max_value = -FLT_MAX;
        int max_id = 0;
        for (int j = 0; j < num_classes; ++j) {
          if (this->blob_bottom_data_->data_at(i, j, h, w) > max_value) {
            max_value = this->blob_bottom_data_->data_at(i, j, h, w);
            max_id = j;
          }
        }
        if (max_id == this->blob_bottom_label_->data_at(i, 0, h, w)) {
          ++num_correct_labels;
        }
      }
    }
  }
  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0),
              num_correct_labels / static_cast<double>(num * height * width),
              1e-4);
}

TYPED_TEST(PixelAccuracyLayerTest, TestForwardCPUTopK) {
  LayerParameter layer_param;
  AccuracyParameter *accuracy_param = layer_param.mutable_accuracy_param();
  accuracy_param->set_top_k(this->top_k_);
  PixelAccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));

  int num_correct_labels = 0;
  const int num = this->blob_bottom_data_->num();
  const int num_classes = this->blob_bottom_data_->channels();
  const int height = this->blob_bottom_data_->height();
  const int width = this->blob_bottom_data_->width();
  for (int i = 0; i < num; ++i) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        for (int j = 0; j < num_classes; ++j) {
          TypeParam current_value =
              this->blob_bottom_data_->data_at(i, j, h, w);
          int current_rank = 0;
          for (int k = 0; k < num_classes; ++k) {
            if (this->blob_bottom_data_->data_at(i, k, h, w) > current_value) {
              ++current_rank;
            }
          }
          if (current_rank < this->top_k_ &&
              j == this->blob_bottom_label_->data_at(i, 0, h, w)) {
            ++num_correct_labels;
          }
        }
      }
    }
  }
  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0),
              num_correct_labels / static_cast<double>(num * height * width),
              1e-4);
}

TYPED_TEST(PixelAccuracyLayerTest, TestForwardCPUIgnoreBg) {
  LayerParameter layer_param;
  LabelParameter *label_param = layer_param.mutable_label_param();
  label_param->set_ignore_label_zero(true);
  // Re-populate bottom_label with +1 more class.
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  shared_ptr<Caffe::RNG> rng(new Caffe::RNG(prefetch_rng_seed));
  caffe::rng_t *prefetch_rng = static_cast<caffe::rng_t *>(rng->generator());
  TypeParam *label_data = this->blob_bottom_label_->mutable_cpu_data();
  for (int i = 0; i < 100 * 5 * 3; ++i) {
    label_data[i] = (*prefetch_rng)() % 11;
  }

  Caffe::set_mode(Caffe::CPU);
  PixelAccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));

  const int num = this->blob_bottom_data_->num();
  const int num_classes = this->blob_bottom_data_->channels();
  const int height = this->blob_bottom_data_->height();
  const int width = this->blob_bottom_data_->width();
  TypeParam num_correct_labels = 0;
  int num_relevant_labels = 0;
  for (int i = 0; i < num; ++i) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
	int pixel_label = this->blob_bottom_label_->data_at(i, 0, h, w);
	if (pixel_label != 0) {
	  // Fix offset.
	  --pixel_label;
	  ++num_relevant_labels;
	  TypeParam max_value = -FLT_MAX;
	  int max_id = 0;
	  for (int j = 0; j < num_classes; ++j) {
	    if (this->blob_bottom_data_->data_at(i, j, h, w) > max_value) {
	      max_value = this->blob_bottom_data_->data_at(i, j, h, w);
	      max_id = j;
	    }
	  }
	  if (max_id == pixel_label) {
	    ++num_correct_labels;
	  }
	}
      }
    }
  }
  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0),
              num_correct_labels / num_relevant_labels, 1e-4);
}

} // namespace caffe
