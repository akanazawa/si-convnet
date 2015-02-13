// Angjoo Kanazawa 2013
#include <cstring>
#include <cuda_runtime.h>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class UpsamplingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

protected:
  UpsamplingLayerTest()
      : blob_bottom_(new Blob<Dtype>()), blob_top_(new Blob<Dtype>()) {};
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 2, 3);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  };
  virtual ~UpsamplingLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype> *const blob_bottom_;
  Blob<Dtype> *const blob_top_;
  vector<Blob<Dtype> *> blob_bottom_vec_;
  vector<Blob<Dtype> *> blob_top_vec_;

  void printMat(const float *data, const int &row, const int &col) {
    for (int i = 0; i < row * col; ++i) {
      printf("%.3f\t", data[i]);
      if ((i + 1) % col == 0)
        printf("\n");
    }
    printf("\n");
  }
};

TYPED_TEST_CASE(UpsamplingLayerTest, TestDtypesAndDevices);

TYPED_TEST(UpsamplingLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_->Reshape(2, 2, 3, 4);
  LayerParameter layer_param;
  TransParameter *t0 = layer_param.add_transformations(); // add identity
  t0->set_interp(NN);
  TransParameter *t1 = layer_param.add_transformations();
  t1->set_scale(2.);
  TransParameter *t2 = layer_param.add_transformations();
  t2->set_rotation(20.);
  Blob<Dtype> *blob_top_1 = new Blob<Dtype>();
  Blob<Dtype> *blob_top_2 = new Blob<Dtype>();
  this->blob_top_vec_.push_back(blob_top_1);
  this->blob_top_vec_.push_back(blob_top_2);

  UpsamplingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));

  // 0th top should be identity.
  EXPECT_EQ(this->blob_top_->count(), this->blob_bottom_->count());
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
  // 1st is 2x scale
  EXPECT_EQ(blob_top_1->num(), this->blob_bottom_->num());
  EXPECT_EQ(blob_top_1->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(blob_top_1->height(), this->blob_bottom_->height() * 2);
  EXPECT_EQ(blob_top_1->width(), this->blob_bottom_->width() * 2);
  // 2nd is..
  EXPECT_EQ(blob_top_2->num(), this->blob_bottom_->num());
  EXPECT_EQ(blob_top_2->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(blob_top_2->height(), 4);
  EXPECT_EQ(blob_top_2->width(), 4);

  delete blob_top_1;
  delete blob_top_2;
}

TYPED_TEST(UpsamplingLayerTest, TestSimpleForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TransParameter *t0 = layer_param.add_transformations(); // add identity
  t0->set_interp(NN);
  TransParameter *t1 = layer_param.add_transformations();
  t1->set_scale(2.);
  TransParameter *t2 = layer_param.add_transformations();
  t2->set_scale(0.5);

  Blob<Dtype> *blob_top_1 = new Blob<Dtype>();
  Blob<Dtype> *blob_top_2 = new Blob<Dtype>();
  this->blob_top_vec_.push_back(blob_top_1);
  this->blob_top_vec_.push_back(blob_top_2);

  this->blob_bottom_->Reshape(2, 3, 2, 2);
  Dtype img[4] = { 0, 1, 1, 0 };

  int width = this->blob_bottom_->width();
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int i = 0; i < 4; ++i) {
        this->blob_bottom_->set_data_at(img[i], n, c, i / width, i % width);
      }
    }
  }

  Dtype want[16] = { 0., 0., 1., 1., 0., 0., 1., 1.,
                     1., 1., 0., 0., 1., 1., 0., 0. };
  UpsamplingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));

  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));

  EXPECT_EQ(this->blob_top_->count(), this->blob_bottom_->count());
  // identity shouldn't change anything
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    ASSERT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
  }
  // top_1 should be 2x
  const int width_top = blob_top_1->width();
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int i = 0; i < 16; ++i) {
        ASSERT_EQ(blob_top_1->data_at(n, c, i / width_top, i % width_top),
                  want[i]);
      }
    }
  }

  // -- setup for bkwd --
  Dtype top_diff[4] = { 10, 10, 10, 10 };
  Dtype top_diff_1[16] = { 100, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0 };
  Dtype top_diff_2[1] = { 0 };

  for (int n = 0; n < this->blob_top_->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int i = 0; i < 4; ++i) {
        this->blob_top_->set_diff_at(top_diff[i], n, c, i / 2, i % 2);
      }
      for (int i = 0; i < 16; ++i) {
        blob_top_1->set_diff_at(top_diff_1[i], n, c, i / 4, i % 4);
      }
      for (int i = 0; i < 1; ++i) {
        blob_top_2->set_diff_at(top_diff_2[i], n, c, i / 1, i % 1);
      }
    }
  }

  // result is sum of sdiff
  Dtype want_diff[4] = { 110, 10, 110, 10 };
  vector<bool> propagate_down(1, true);
  layer.Backward(this->blob_top_vec_, propagate_down,
                 &(this->blob_bottom_vec_));

  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int i = 0; i < 4; ++i) {
        ASSERT_EQ(this->blob_bottom_->diff_at(n, c, i / 2, i % 2),
                  want_diff[i]);
      }
    }
  }

  delete blob_top_1;
  delete blob_top_2;
}

TYPED_TEST(UpsamplingLayerTest, TestSimpleForwardWithCrop) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TransParameter *t0 = layer_param.add_transformations(); // add identity
  t0->set_interp(BILINEAR);
  TransParameter *t1 = layer_param.add_transformations();
  t1->set_scale(2.);
  t1->set_final_width(2);
  t1->set_final_height(2);
  TransParameter *t2 = layer_param.add_transformations();
  t2->set_scale(0.5);
  t2->set_final_width(2);
  t2->set_final_height(2);

  Blob<Dtype> *blob_top_1 = new Blob<Dtype>();
  Blob<Dtype> *blob_top_2 = new Blob<Dtype>();
  this->blob_top_vec_.push_back(blob_top_1);
  this->blob_top_vec_.push_back(blob_top_2);

  this->blob_bottom_->Reshape(2, 3, 2, 2);
  Dtype img[4] = { 0, 1, 1, 0 };

  int width = this->blob_bottom_->width();
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int i = 0; i < 4; ++i) {
        this->blob_bottom_->set_data_at(img[i], n, c, i / width, i % width);
      }
    }
  }

  UpsamplingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));

  // Identity.
  EXPECT_EQ(this->blob_top_->count(), this->blob_bottom_->count());
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    ASSERT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
  }

  // x2.
  const Dtype want[4] = { 0.375, 0.625, 0.625, 0.375 };
  EXPECT_EQ(blob_top_1->width(), 2);
  EXPECT_EQ(blob_top_1->height(), 2);
  const int width_top1 = blob_top_1->width();
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int i = 0; i < width_top1 * width_top1; ++i) {
        ASSERT_EQ(blob_top_1->data_at(n, c, i / width_top1, i % width_top1),
                  want[i]);
        // printf("%.3f\t", blob_top_1->data_at(n, c, i / width_top1, i %
        // width_top1));
        // if (((i+1) % width_top1) == 0) printf("\n");
      }
      // printf("\n");
    }
  }

  // x0.5. All 0
  const Dtype want2[4] = { 0, 0, 0, 0.5 };
  EXPECT_EQ(blob_top_2->width(), 2);
  EXPECT_EQ(blob_top_2->height(), 2);
  const int width_top = blob_top_2->width();
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int i = 0; i < width_top * width_top; ++i) {
        ASSERT_EQ(blob_top_2->data_at(n, c, i / width_top, i % width_top),
                  want2[i]);
        // printf("%.3f\t", blob_top_2->data_at(n, c, i / width_top, i %
        // width_top));
        // if (((i+1) % width_top) == 0) printf("\n");
      }
    }
  }

  delete blob_top_1;
  delete blob_top_2;
}

TYPED_TEST(UpsamplingLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_->Reshape(2, 2, 6, 6);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);

  LayerParameter layer_param;
  layer_param.add_transformations(); // add identity
  TransParameter *t1 = layer_param.add_transformations();
  t1->set_scale(2.);
  TransParameter *t2 = layer_param.add_transformations();
  t2->set_scale(.5);
  TransParameter *t3 = layer_param.add_transformations();
  t3->set_rotation(45);

  Blob<Dtype> *blob_top_1 = new Blob<Dtype>();
  Blob<Dtype> *blob_top_2 = new Blob<Dtype>();
  Blob<Dtype> *blob_top_3 = new Blob<Dtype>();

  this->blob_top_vec_.push_back(blob_top_1);
  this->blob_top_vec_.push_back(blob_top_2);
  this->blob_top_vec_.push_back(blob_top_3);

  UpsamplingLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
                                  &(this->blob_top_vec_));
  delete blob_top_1;
  delete blob_top_2;
  delete blob_top_3;
}

TYPED_TEST(UpsamplingLayerTest, TestGradient_bilinear) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_->Reshape(2, 2, 6, 6);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);

  LayerParameter layer_param;
  TransParameter *t0 = layer_param.add_transformations(); // add identity
  t0->set_interp(BILINEAR);
  TransParameter *t1 = layer_param.add_transformations();
  t1->set_scale(2.);
  TransParameter *t2 = layer_param.add_transformations();
  t2->set_scale(.5);
  TransParameter *t3 = layer_param.add_transformations();
  t3->set_rotation(45);

  Blob<Dtype> *blob_top_1 = new Blob<Dtype>();
  Blob<Dtype> *blob_top_2 = new Blob<Dtype>();
  Blob<Dtype> *blob_top_3 = new Blob<Dtype>();

  this->blob_top_vec_.push_back(blob_top_1);
  this->blob_top_vec_.push_back(blob_top_2);
  this->blob_top_vec_.push_back(blob_top_3);

  UpsamplingLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
                                  &(this->blob_top_vec_));
  delete blob_top_1;
  delete blob_top_2;
  delete blob_top_3;
}
}
