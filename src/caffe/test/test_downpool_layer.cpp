// Angjoo Kanazawa 2013
#include <cstring>

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
class DownPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

protected:
  // Multiple bottoms, one top.
  DownPoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>()), blob_top_(new Blob<Dtype>()) {};
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 2, 2);
    // Fill the values.
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  };
  virtual ~DownPoolingLayerTest() {
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

TYPED_TEST_CASE(DownPoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(DownPoolingLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_->Reshape(2, 2, 3, 3);
  LayerParameter layer_param;
  layer_param.add_transformations(); // add identity
  TransParameter *t1 = layer_param.add_transformations();
  t1->set_scale(1 / 2.);
  TransParameter *t2 = layer_param.add_transformations();
  // t2->set_rotation(20.);
  t2->set_scale(0.75);
  Blob<Dtype> *blob_bottom_1 = new Blob<Dtype>(2, 2, 6, 8);
  Blob<Dtype> *blob_bottom_2 = new Blob<Dtype>(2, 2, 4, 6);
  this->blob_bottom_vec_.push_back(blob_bottom_1);
  this->blob_bottom_vec_.push_back(blob_bottom_2);

  DownPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));

  // Only one top!
  EXPECT_EQ(this->blob_top_vec_.size(), 1);
  // and its size should be identity (equal to the first bottom layer)
  EXPECT_EQ(this->blob_top_->count(), this->blob_bottom_->count());
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());

  delete blob_bottom_1;
  delete blob_bottom_2;
}

TYPED_TEST(DownPoolingLayerTest, TestSimpleForwardBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TransParameter *t0 = layer_param.add_transformations();
  t0->set_interp(NN);
  TransParameter *t1 = layer_param.add_transformations();
  t1->set_scale(2.);
  TransParameter *t2 = layer_param.add_transformations();
  t2->set_scale(4);
  TransParameter *t3 = layer_param.add_transformations();
  t3->set_rotation(-90);
  // 2x2, 4x4, 8x8, and 2 x 2 with:
  // [0 1;  [0 0;  [0 0;  [0 4;
  //  1 1]   2 2]   0 3];  0 0];
  // when all are transforemd into canonical shape of 2x2

  // bottom 0, the canonical one:
  this->blob_bottom_->Reshape(2, 3, 2, 2);
  Dtype img[4] = { 1, 1, 1, 1 };
  int width = this->blob_bottom_->width();
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int i = 0; i < 4; ++i) {
        this->blob_bottom_->set_data_at(img[i], n, c, i / width, i % width);
        // printf("%.1f ", this->blob_bottom_->data_at( n, c, i / width, i %
        // width));
        // if (i % 2) printf("\n");
      }
    }
  }
  // bottom 1, 4x4:
  Blob<Dtype> *blob_bottom_1 = new Blob<Dtype>(2, 3, 4, 4);
  Dtype img1[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2 };
  width = blob_bottom_1->width();
  for (int n = 0; n < blob_bottom_1->num(); ++n) {
    for (int c = 0; c < blob_bottom_1->channels(); ++c) {
      for (int i = 0; i < 16; ++i) {
        blob_bottom_1->set_data_at(img1[i], n, c, i / width, i % width);
        // printf("%.1f ", this->blob_bottom_1->data_at( n, c, i / width, i %
        // width));
        // if (i % 2) printf("\n");
      }
    }
  }
  // bottom 2, 8x8
  Blob<Dtype> *blob_bottom_2 = new Blob<Dtype>(2, 3, 8, 8);
  Dtype img2[64] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0, 3, 3, 3, 3,
                     0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0, 3, 3, 3, 3 };
  width = blob_bottom_2->width();
  for (int n = 0; n < blob_bottom_2->num(); ++n) {
    for (int c = 0; c < blob_bottom_2->channels(); ++c) {
      for (int i = 0; i < 64; ++i) {
        blob_bottom_2->set_data_at(img2[i], n, c, i / width, i % width);
        // printf("%.1f ", this->blob_bottom_2->data_at( n, c, i / width, i %
        // width));
        // if (i % 2) printf("\n");
      }
    }
  }

  // bottom3  2x2
  Blob<Dtype> *blob_bottom_3 = new Blob<Dtype>(2, 3, 2, 2);
  Dtype img3[4] = { 0, 0, 0, 4 };
  for (int n = 0; n < blob_bottom_3->num(); ++n) {
    for (int c = 0; c < blob_bottom_3->channels(); ++c) {
      for (int i = 0; i < 4; ++i) {
        blob_bottom_3->set_data_at(img3[i], n, c, i / 2, i % 2);
        // printf("%.1f ", this->blob_bottom_3->data_at( n, c, i / width, i %
        // width));
        // if (i % 2) printf("\n");
      }
    }
  }

  this->blob_bottom_vec_.push_back(blob_bottom_1);
  this->blob_bottom_vec_.push_back(blob_bottom_2);
  this->blob_bottom_vec_.push_back(blob_bottom_3);
  // want for top:
  Dtype want[4] = { 1, 4, 2, 3 };
  //----------Start----------
  DownPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));

  //----------Forward----------
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));

  EXPECT_EQ(this->blob_top_->count(), this->blob_bottom_->count());

  // this->printMat(this->blob_top_->cpu_data(), this->blob_top_->height(),
  // this->blob_top_->width());
  // Top should be same as want
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int i = 0; i < 4; ++i) {
        // printf("%.1f vs %.1f (n%d c%d i%d)\t",
        //        this->blob_top_->data_at( n, c, i / 2, i % 2), want[i],
        //        n, c, i);
        // if (i % 2) printf("\n");
        ASSERT_EQ(this->blob_top_->data_at(n, c, i / 2, i % 2), want[i]);
      }
    }
  }

  // -- setup for bkwd --
  Dtype top_diff[4] = { 100, 10, 20, 30 };

  for (int n = 0; n < this->blob_top_->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int i = 0; i < 4; ++i) {
        this->blob_top_->set_diff_at(top_diff[i], n, c, i / 2, i % 2);
        // printf("%.1f ", this->blob_top_->data_at( n, c, i / 2, i % 2));
        // if (i % 2) printf("\n");
      }
    }
  }

  Dtype want_diff[4] = { top_diff[0], 0, 0, 0 };
  Dtype want_diff_1[16] = { 0, 0, 0, 0, 0, 0,           0, 0,
                            0, 0, 0, 0, 0, top_diff[2], 0, 0 };

  Dtype want_diff_2[64] = { 0, 0, 0,           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0,           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0,           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0,           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, top_diff[3], 0, 0, 0, 0, 0, 0, 0, 0, 0 };

  Dtype want_diff_3[4] = { 0, 0, 0, top_diff[1] };
  // print switch_idx_
  // this->printMat(layer.max_switch().cpu_data(), layer.max_switch().height(),
  // layer.max_switch().width());

  //----------Backward----------
  vector<bool> propagate_down(1, true);
  layer.Backward(this->blob_top_vec_, propagate_down,
                 &(this->blob_bottom_vec_));

  // bottom 0:
  // this->printMat(this->blob_bottom_->cpu_diff(), 2, 2);
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int i = 0; i < 4; ++i) {
        ASSERT_EQ(this->blob_bottom_->diff_at(n, c, i / 2, i % 2),
                  want_diff[i]);
      }
    }
  }

  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int i = 0; i < 16; ++i) {
        ASSERT_EQ(blob_bottom_1->diff_at(n, c, i / 4, i % 4), want_diff_1[i]);
      }
    }
  }

  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int i = 0; i < 64; ++i) {
        ASSERT_EQ(blob_bottom_2->diff_at(n, c, i / 8, i % 8), want_diff_2[i]);
      }
    }
  }

  // this->printMat(this->blob_bottom_->cpu_diff(), 2, 2);
  // this->printMat(blob_bottom_1->cpu_diff(), 4, 4);
  // this->printMat(blob_bottom_3->cpu_diff(), 2, 2);
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int i = 0; i < 4; ++i) {
        ASSERT_EQ(blob_bottom_3->diff_at(n, c, i / 2, i % 2), want_diff_3[i]);
      }
    }
  }

  delete blob_bottom_1;
  delete blob_bottom_2;
  delete blob_bottom_3;
}

TYPED_TEST(DownPoolingLayerTest, TestSimpleForwardBackwardBilinear) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TransParameter *t0 = layer_param.add_transformations();
  t0->set_interp(BILINEAR);
  TransParameter *t1 = layer_param.add_transformations();
  t1->set_scale(2.);
  TransParameter *t2 = layer_param.add_transformations();
  t2->set_scale(4);
  TransParameter *t3 = layer_param.add_transformations();
  t3->set_rotation(-90);
  // 2x2, 4x4, 8x8, and 2 x 2 with:
  // [0 1;  [0 0;  [0 0;  [0 4;
  //  1 1]   2 2]   0 3];  0 0];
  // when all are transforemd into canonical shape of 2x2

  // bottom 0, the canonical one:
  this->blob_bottom_->Reshape(2, 3, 2, 2);
  Dtype img[4] = { 1, 1, 1, 1 };
  int width = this->blob_bottom_->width();
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int i = 0; i < 4; ++i) {
        this->blob_bottom_->set_data_at(img[i], n, c, i / width, i % width);
      }
    }
  }
  // bottom 1, 4x4:
  Blob<Dtype> *blob_bottom_1 = new Blob<Dtype>(2, 3, 4, 4);
  Dtype img1[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2 };
  width = blob_bottom_1->width();
  for (int n = 0; n < blob_bottom_1->num(); ++n) {
    for (int c = 0; c < blob_bottom_1->channels(); ++c) {
      for (int i = 0; i < 16; ++i) {
        blob_bottom_1->set_data_at(img1[i], n, c, i / width, i % width);
      }
    }
  }
  // bottom 2, 8x8
  Blob<Dtype> *blob_bottom_2 = new Blob<Dtype>(2, 3, 8, 8);
  Dtype img2[64] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0, 3, 3, 3, 3,
                     0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0, 3, 3, 3, 3 };
  width = blob_bottom_2->width();
  for (int n = 0; n < blob_bottom_2->num(); ++n) {
    for (int c = 0; c < blob_bottom_2->channels(); ++c) {
      for (int i = 0; i < 64; ++i) {
        blob_bottom_2->set_data_at(img2[i], n, c, i / width, i % width);
      }
    }
  }

  // bottom3  2x2
  Blob<Dtype> *blob_bottom_3 = new Blob<Dtype>(2, 3, 2, 2);
  Dtype img3[4] = { 0, 0, 0, 4 };
  for (int n = 0; n < blob_bottom_3->num(); ++n) {
    for (int c = 0; c < blob_bottom_3->channels(); ++c) {
      for (int i = 0; i < 4; ++i) {
        blob_bottom_3->set_data_at(img3[i], n, c, i / 2, i % 2);
      }
    }
  }

  this->blob_bottom_vec_.push_back(blob_bottom_1);
  this->blob_bottom_vec_.push_back(blob_bottom_2);
  this->blob_bottom_vec_.push_back(blob_bottom_3);

  Dtype want[4] = { 1, 4, 2, 3 };
  //----------Start----------
  DownPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));

  //----------Forward----------
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));

  EXPECT_EQ(this->blob_top_->count(), this->blob_bottom_->count());

  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int i = 0; i < 4; ++i) {
        ASSERT_EQ(this->blob_top_->data_at(n, c, i / 2, i % 2), want[i]);
      }
    }
  }
  // -- setup for bkwd --
  Dtype top_diff[4] = { 100, 10, 20, 30 };

  for (int n = 0; n < this->blob_top_->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int i = 0; i < 4; ++i) {
        this->blob_top_->set_diff_at(top_diff[i], n, c, i / 2, i % 2);
      }
    }
  }

  Dtype want_diff[4] = { top_diff[0], 0, 0, 0 };
  float val = top_diff[2] * 0.25;
  Dtype want_diff_1[16] = { 0,   0,   0, 0, 0,   0,   0, 0,
                            val, val, 0, 0, val, val, 0, 0 };
  val = top_diff[3] * 0.25;
  Dtype want_diff_2[64] = { 0, 0,   0,   0, 0, 0, 0,   0,   0, 0, 0, 0, 0,
                            0, 0,   0,   0, 0, 0, 0,   0,   0, 0, 0, 0, 0,
                            0, 0,   0,   0, 0, 0, 0,   0,   0, 0, 0, 0, 0,
                            0, 0,   0,   0, 0, 0, val, val, 0, 0, 0, 0, 0,
                            0, val, val, 0, 0, 0, 0,   0,   0, 0, 0, 0 };

  Dtype want_diff_3[4] = { 0, 0, 0, top_diff[1] };

  //----------Backward----------
  vector<bool> propagate_down(1, true);
  layer.Backward(this->blob_top_vec_, propagate_down,
                 &(this->blob_bottom_vec_));

  // bottom 0:
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int i = 0; i < 4; ++i) {
        ASSERT_EQ(this->blob_bottom_->diff_at(n, c, i / 2, i % 2),
                  want_diff[i]);
      }
    }
  }

  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int i = 0; i < 16; ++i) {
        ASSERT_EQ(blob_bottom_1->diff_at(n, c, i / 4, i % 4), want_diff_1[i]);
      }
    }
  }

  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int i = 0; i < 64; ++i) {
        ASSERT_EQ(blob_bottom_2->diff_at(n, c, i / 8, i % 8), want_diff_2[i]);
      }
    }
  }

  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int i = 0; i < 4; ++i) {
        ASSERT_EQ(blob_bottom_3->diff_at(n, c, i / 2, i % 2), want_diff_3[i]);
      }
    }
  }

  delete blob_bottom_1;
  delete blob_bottom_2;
  delete blob_bottom_3;
}

// Test that padded region in downpool never gets max switch index
TYPED_TEST(DownPoolingLayerTest, TestValidProp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TransParameter *t0 = layer_param.add_transformations();
  t0->set_interp(BILINEAR);
  TransParameter *t1 = layer_param.add_transformations();
  t1->set_scale(0.5);
  TransParameter *t2 = layer_param.add_transformations();
  t2->set_scale(2);

  // result of input size 6x6 with kernel 2x2 with identity, 0.5, 2 scale:
  // conv result is 5x5, 2x2, 11x11. After inverse transformation:
  // 5x5, 4x4, 5x5, where 4x4 output is 0-padded so that it's 5x5.
  // we need to make sure that 0 padded region never wins the max
  int num = 4;
  int channel = 10;
  // bottom 0, the canonical one:
  this->blob_bottom_->Reshape(num, channel, 5, 5);
  // bottom 1, the one with padding
  Blob<Dtype> *blob_bottom_1 = new Blob<Dtype>(num, channel, 2, 2);
  // fill this with the largest value
  for (int n = 0; n < blob_bottom_1->num(); ++n) {
    for (int c = 0; c < blob_bottom_1->channels(); ++c) {
      for (int i = 0; i < 4; ++i) {
        blob_bottom_1->set_data_at(100, n, c, i / 2, i % 2);
      }
    }
  }

  // bottom 2, smaller values than bottom 1
  Blob<Dtype> *blob_bottom_2 = new Blob<Dtype>(num, channel, 11, 11);
  for (int n = 0; n < blob_bottom_1->num(); ++n) {
    for (int c = 0; c < blob_bottom_1->channels(); ++c) {
      for (int i = 0; i < 121; ++i) {
        blob_bottom_2->set_data_at(2, n, c, i / 11, i % 11);
      }
    }
  }

  this->blob_bottom_vec_.push_back(blob_bottom_1);
  this->blob_bottom_vec_.push_back(blob_bottom_2);

  // make layer
  DownPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // do forward
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  const Blob<float> &max_switch = layer.max_switch();
  // printf("max_switch:\n");
  // this->printMat(max_switch.cpu_data(), this->blob_top_->height(),
  // this->blob_top_->width());
  // const vector<shared_ptr<Blob<float> > > &coord_idx = layer.coord_idx();
  // printf("coord_idx 1:\n");
  // this->printMat(coord_idx[1]->cpu_data(), this->blob_top_->height(),
  // this->blob_top_->width());
  // printf("coord_idx 2:\n");
  // this->printMat(coord_idx[2]->cpu_data(), this->blob_top_->height(),
  // this->blob_top_->width());

  Dtype want_switch[25] = { 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1,
                            1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1 };

  Dtype want[25] = { 2,   2,   2,   2,   2,   2,   100, 100, 100,
                     100, 2,   100, 100, 100, 100, 2,   100, 100,
                     100, 100, 2,   100, 100, 100, 100 };

  // this->printMat(this->blob_top_->cpu_data(), this->blob_top_->height(),
  // this->blob_top_->width());
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int i = 0; i < 25; ++i) {
        ASSERT_EQ(this->blob_top_->data_at(n, c, i / 5, i % 5), want[i]);
      }
    }
  }
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int i = 0; i < 25; ++i) {
        ASSERT_EQ(max_switch.data_at(n, c, i / 5, i % 5), want_switch[i]);
      }
    }
  }
}

// Note this won't pass if all inputs are 0 bc of the max pooling nature, when
// -delta x is applied to the input to compute numerical gradient, that change
// won't survive
// Bottomline: do fill the bottom matrices.
TYPED_TEST(DownPoolingLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_->Reshape(2, 2, 7, 7);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);

  LayerParameter layer_param;
  TransParameter *t0 = layer_param.add_transformations(); // add identity
  t0->set_interp(NN);
  TransParameter *t1 = layer_param.add_transformations();
  t1->set_scale(2.);
  TransParameter *t2 = layer_param.add_transformations();
  t2->set_scale(.5);
  TransParameter *t3 = layer_param.add_transformations();
  t3->set_rotation(45);
  // // bottom 1, 4x4:
  Blob<Dtype> *blob_bottom_1 = new Blob<Dtype>(2, 2, 14, 14);
  Blob<Dtype> *blob_bottom_2 = new Blob<Dtype>(2, 2, 3, 3);
  Blob<Dtype> *blob_bottom_3 = new Blob<Dtype>(2, 2, 5, 5);

  filler.Fill(blob_bottom_1);
  filler.Fill(blob_bottom_2);
  filler.Fill(blob_bottom_3);
  this->blob_bottom_vec_.push_back(blob_bottom_1);
  this->blob_bottom_vec_.push_back(blob_bottom_2);
  this->blob_bottom_vec_.push_back(blob_bottom_3);

  DownPoolingLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
                                  &(this->blob_top_vec_));

  delete blob_bottom_1;
  delete blob_bottom_2;
  delete blob_bottom_3;
}

TYPED_TEST(DownPoolingLayerTest, TestGradientBilinear) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_->Reshape(2, 2, 7, 7);
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
  // // bottom 1, 4x4:
  Blob<Dtype> *blob_bottom_1 = new Blob<Dtype>(2, 2, 14, 14);
  Blob<Dtype> *blob_bottom_2 = new Blob<Dtype>(2, 2, 3, 3);
  Blob<Dtype> *blob_bottom_3 = new Blob<Dtype>(2, 2, 5, 5);

  filler.Fill(blob_bottom_1);
  filler.Fill(blob_bottom_2);
  filler.Fill(blob_bottom_3);
  this->blob_bottom_vec_.push_back(blob_bottom_1);
  this->blob_bottom_vec_.push_back(blob_bottom_2);
  this->blob_bottom_vec_.push_back(blob_bottom_3);

  DownPoolingLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
                                  &(this->blob_top_vec_));

  delete blob_bottom_1;
  delete blob_bottom_2;
  delete blob_bottom_3;
}
}
