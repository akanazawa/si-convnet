#include <cstring>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class BlobSimpleTest : public ::testing::Test {
 protected:
  BlobSimpleTest()
      : blob_(new Blob<Dtype>()),
        blob_preshaped_(new Blob<Dtype>(2, 3, 4, 5)),
	blob_nodiff_(new Blob<Dtype>(false)),
	blob_preshaped_nodiff_(new Blob<Dtype>(2, 3, 4, 5, false)) {}
  virtual ~BlobSimpleTest() { 
    delete blob_; delete blob_preshaped_; 
    delete blob_nodiff_; delete blob_preshaped_nodiff_; 
  }
  Blob<Dtype>* const blob_;
  Blob<Dtype>* const blob_nodiff_;
  Blob<Dtype>* const blob_preshaped_;
  Blob<Dtype>* const blob_preshaped_nodiff_;
};

typedef ::testing::Types<float> Dtypes;
// typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(BlobSimpleTest, Dtypes);

TYPED_TEST(BlobSimpleTest, TestInitialization) {
  EXPECT_TRUE(this->blob_);
  EXPECT_TRUE(this->blob_preshaped_);
  EXPECT_EQ(this->blob_preshaped_->num(), 2);
  EXPECT_EQ(this->blob_preshaped_->channels(), 3);
  EXPECT_EQ(this->blob_preshaped_->height(), 4);
  EXPECT_EQ(this->blob_preshaped_->width(), 5);
  EXPECT_EQ(this->blob_preshaped_->count(), 120);
  EXPECT_EQ(this->blob_->num(), 0);
  EXPECT_EQ(this->blob_->channels(), 0);
  EXPECT_EQ(this->blob_->height(), 0);
  EXPECT_EQ(this->blob_->width(), 0);
  EXPECT_EQ(this->blob_->count(), 0);
  EXPECT_EQ(this->blob_->allocateDiff(), true);

  EXPECT_TRUE(this->blob_nodiff_);
  EXPECT_TRUE(this->blob_preshaped_nodiff_);
  EXPECT_EQ(this->blob_preshaped_nodiff_->num(), 2);
  EXPECT_EQ(this->blob_preshaped_nodiff_->channels(), 3);
  EXPECT_EQ(this->blob_preshaped_nodiff_->height(), 4);
  EXPECT_EQ(this->blob_preshaped_nodiff_->width(), 5);
  EXPECT_EQ(this->blob_preshaped_nodiff_->count(), 120);
  EXPECT_EQ(this->blob_preshaped_nodiff_->allocateDiff(), false);
  EXPECT_EQ(this->blob_nodiff_->num(), 0);
  EXPECT_EQ(this->blob_nodiff_->channels(), 0);
  EXPECT_EQ(this->blob_nodiff_->height(), 0);
  EXPECT_EQ(this->blob_nodiff_->width(), 0);
  EXPECT_EQ(this->blob_nodiff_->count(), 0);
  EXPECT_EQ(this->blob_nodiff_->allocateDiff(), false);

}

TYPED_TEST(BlobSimpleTest, TestPointersCPUGPU) {
  EXPECT_TRUE(this->blob_preshaped_->gpu_data());
  EXPECT_TRUE(this->blob_preshaped_->cpu_data());
  EXPECT_TRUE(this->blob_preshaped_->mutable_gpu_data());
  EXPECT_TRUE(this->blob_preshaped_->mutable_cpu_data());
}

TYPED_TEST(BlobSimpleTest, TestReshape) {
  this->blob_->Reshape(2, 3, 4, 5);
  EXPECT_EQ(this->blob_->num(), 2);
  EXPECT_EQ(this->blob_->channels(), 3);
  EXPECT_EQ(this->blob_->height(), 4);
  EXPECT_EQ(this->blob_->width(), 5);
  EXPECT_EQ(this->blob_->count(), 120);
}

}  // namespace caffe
