// Angjoo Kanazawa 2013
#include "gtest/gtest.h"
#include "caffe/syncedmem.hpp"
#include "caffe/util/transformation.hpp"
#include "caffe/util/imshow.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <cmath>
#include <vector>

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

//----------basic functions----------
TEST(AddTransform, identity) {
  float orig[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
  float tmp[9] = { 3, 0, 0, 0, 1, 0, 0, 0, 1 };
  float *want = tmp;
  AddTransform(orig, tmp, RIGHT);
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(want[i], orig[i]);
  }
}

TEST(AddTransform, test_left) {
  float A[9] = { 0, 1, 6, 3, 5, 7, 4, 0, 2 };
  float B[9] = { 1, 0, 0, 4, 1, 4, 0, 4, 2 };
  float want[9] = { 0, 1, 6, 19, 9, 39, 20, 20, 32 };
  AddTransform(A, B, LEFT);
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(want[i], A[i]);
  }
}

TEST(AddTransform, square) {
  float orig[9] = { 8, 1, 6, 3, 5, 7, 4, 9, 2 };
  float tmp[9] = { 8, 1, 6, 3, 5, 7, 4, 9, 2 };
  float want[9] = { 91, 67, 67, 67, 91, 67, 67, 67, 91 };
  AddTransform(orig, tmp, RIGHT);
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(want[i], orig[i]);
  }
}

TEST(AddTransform, orthogonal) {
  float rad = 4.5;
  float orig[9] = { cos(rad), -sin(rad), 0, sin(rad), cos(rad), 0, 0, 0, 1 };
  float tmp[9] = { cos(rad), sin(rad), 0, -sin(rad), cos(rad), 0, 0, 0, 1 };
  float want[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
  AddTransform(orig, tmp, RIGHT);
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(want[i], orig[i]);
  }
}

TEST(AddTransform, rotation) {
  float rad = 0;
  float orig[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
  float tmp[9] = { cos(rad), sin(rad), 0, -sin(rad), cos(rad), 0, 0, 0, 1 };
  float want[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
  AddTransform(orig, tmp, RIGHT);
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(want[i], orig[i]);
  }
}

TEST(TMatFromProto, scale) {
  TransParameter param;
  float tmat[9];
  float scale = 0.5f;
  param.set_scale(0.5f);
  float want[9] = { scale, 0, 0, 0, scale, 0, 0, 0, 1 };
  TMatFromProto(param, tmat);
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(want[i], tmat[i]);
  }
}

TEST(TMatFromProto, identity) {
  TransParameter param;
  float tmat[9];
  float want[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
  TMatFromProto(param, tmat);
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(want[i], tmat[i]);
  }
}

TEST(TMatToCanonical, scale) {
  TransParameter param;
  float tmat[9];
  param.set_scale(2.f);
  int cano_h = 21;
  int orig_h = 32;
  float scale_want = static_cast<float>(cano_h) / orig_h;
  float want[9] = { scale_want, 0, 0, 0, scale_want, 0, 0, 0, 1 };

  TMatToCanonical(param, cano_h, orig_h, tmat);
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(tmat[i], want[i]);
  }
}

TEST(Invert3x3, identity) {
  float A[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
  float want[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
  Invert3x3(A);
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(want[i], A[i]);
  }
}

TEST(Invert3x3, rotation) {
  float rad = 30 * PI_F / 180;
  float A[9] = { cos(rad), sin(rad), 0, -sin(rad), cos(rad), 0, 0, 0, 1 };
  float want[9] = { cos(rad), -sin(rad), 0, sin(rad), cos(rad), 0, 0, 0, 1 };
  Invert3x3(A);
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(want[i], A[i]);
  }
}

TEST(Invert3x3, scale) {
  float scale = 2.f;
  float A[9] = { scale, 0, 0, 0, scale, 0, 0, 0, 1 };
  float want[9] = { 1 / scale, 0, 0, 0, 1 / scale, 0, 0, 0, 1 };
  Invert3x3(A);
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(want[i], A[i]);
  }
}

TEST(Invert3x3, translation) {
  float dx = .5f;
  float dy = 2.f;
  float A[9] = { 1, 0, 0, 0, 1, 0, dx, dy, 1 };
  float want[9] = { 1, 0, 0, 0, 1, 0, -dx, -dy, 1 };
  Invert3x3(A);
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(want[i], A[i]);
  }
}
TEST(Invert3x3, random) {
  float A[9] = { 2, 1, 1, 1, 1, 3, 3, 1, 3 };
  float want[9] = { 0, -.5, .5, 1.5, 0.75, -1.25, -0.5, 0.25, 0.25 };
  Invert3x3(A);
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(want[i], A[i]);
  }
}

TEST(Reflect, all) {
  // no change
  float val = 5;
  Reflect(val, 10);
  EXPECT_EQ(val, 5);
  // too left
  val = -5;
  Reflect(val, 10);
  EXPECT_EQ(val, 5);

  val = -1;
  Reflect(val, 10);
  EXPECT_EQ(val, 1);

  val = -19;
  Reflect(val, 10);
  EXPECT_EQ(val, 1);

  val = -25;
  Reflect(val, 10);
  EXPECT_EQ(val, 7);
  // too right
  val = 11;
  Reflect(val, 10);
  EXPECT_EQ(val, 7);
}

TEST(Clamp, all) {
  float val = 5;
  Clamp(val, 10);
  EXPECT_EQ(val, 5);

  val = 15;
  Clamp(val, 10);
  EXPECT_EQ(val, 9);

  val = 10;
  Clamp(val, 10);
  EXPECT_EQ(val, 9);

  val = -5;
  Clamp(val, 10);
  EXPECT_EQ(val, 0);
}

TEST(GenBasicCoordMat, square) {
  int w = 2, h = 2;
  float coord[w * h * 3];
  float want[2 * 2 * 3] = { 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1 };

  GenBasicCoordMat(coord, w, h);
  for (int i = 0; i < w * h * 3; ++i) {
    EXPECT_EQ(want[i], coord[i]);
  }
}
TEST(GenBasicCoordMat, rectwide) {
  int w = 3, h = 2;
  float coord[w * h * 3];
  float want[3 * 2 * 3] = {
    0, 0, 1, 0, 1, 1, 0, 2, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1
  };

  GenBasicCoordMat(coord, w, h);
  for (int i = 0; i < w * h * 3; ++i) {
    EXPECT_EQ(want[i], coord[i]);
  }
}

TEST(GenBasicCoordMat, recttall) {
  int w = 2, h = 3;
  float coord[w * h * 3];
  float want[2 * 3 * 3] = {
    0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 2, 0, 1, 2, 1, 1
  };

  GenBasicCoordMat(coord, w, h);
  for (int i = 0; i < w * h * 3; ++i) {
    EXPECT_EQ(want[i], coord[i]);
  }
}

TEST(AddingTransformations, AddRotation) {
  float tmat[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
  float angle = 90.;
  float rad = angle * PI_F / 180;
  float want[9] = { cos(rad), sin(rad), 0, -sin(rad), cos(rad), 0, 0, 0, 1 };
  AddRotation(angle, tmat);
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(want[i], tmat[i]);
  }
}

TEST(AddingTransformations, AddScale) {
  float tmat[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
  float scale = 5.f;
  float want[9] = { scale, 0, 0, 0, scale, 0, 0, 0, 1 };
  AddScale(scale, tmat);
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(want[i], tmat[i]);
  }
}

TEST(AddingTransformations, AddTranslation) {
  float tmat[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
  float dx = 0.5f;
  float dy = 0.8f;
  float want[9] = { 1, 0, 0, 0, 1, 0, dx, dy, 1 };
  AddTranslation(dx, dy, tmat);
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(want[i], tmat[i]);
  }
}

TEST(AddingTransformations, AddTranslation_left) {
  float tmat[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
  float dx = -335.5f;
  float dy = -345.5f;
  float want[9] = { 1, 0, 0, 0, 1, 0, dx, dy, 1 };
  AddTranslation(dx, dy, tmat, LEFT);
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(want[i], tmat[i]);
  }
}

TEST(AddingTransformations, AddScale_Rotation) {
  float tmat[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
  float scale = 0.5f;
  float angle = 15;
  float want[9] = { 0.4830, 0.1294, 0, -0.1294, 0.4830, 0, 0, 0, 1 };
  AddScale(scale, tmat);
  AddRotation(angle, tmat);
  for (int i = 0; i < 9; ++i) {
    ASSERT_TRUE(tmat[i] - want[i] < 0.0001);
  }
}
TEST(AddingTransformations, AddScale_Rotation_translation) {
  float tmat[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
  float scale = 0.5f;
  float angle = 15;
  float dx = 0.25;
  float dy = 1.4;
  float want[9] = { 0.4830, 0.1294, 0, -0.1294, 0.4830, 0, 0.25, 1.4, 1 };
  AddScale(scale, tmat);
  AddRotation(angle, tmat);
  AddTranslation(dx, dy, tmat);
  for (int i = 0; i < 9; ++i) {
    ASSERT_TRUE(tmat[i] - want[i] < 0.0001);
  }
}

TEST(GetNewSize, GetNewSize_scale) {
  float tmat[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
  int h = 50;
  int w = 100;
  AddScale(0.5f, tmat);
  int h_new, w_new;
  int want_h = 25, want_w = 50;

  GetNewSize(h, w, tmat, h_new, w_new);
  ASSERT_EQ(want_h, h_new);
  ASSERT_EQ(want_w, w_new);
}
TEST(GetNewSize, GetNewSize_scale_rot) {
  float tmat[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
  int h = 50;
  int w = 100;
  int angle = 25;
  AddScale(0.5f, tmat);
  AddRotation(angle, tmat);
  int h_new, w_new;

  int want_h = 43, want_w = 55;
  GetNewSize(h, w, tmat, h_new, w_new);
  ASSERT_EQ(want_h, h_new);
  ASSERT_EQ(want_w, w_new);
}

typedef ::testing::Types<float> Dtype;
// typedef ::testing::Types<float, double> Dtype;

// typed Text fixtures, this piece of code is shared by all tests in a testcase
// (independently/no side effect)
template <typename Dtype> class TransformationTest : public ::testing::Test {
protected:
  // called before every test
  virtual void SetUp() {
    coord_ = new Blob<float>(false);
    orig_ = new Blob<Dtype>();
    warped_ = new Blob<Dtype>();
    border_ = CROP;
    interp_ = NN;
    height_ = 3;
    width_ = 3;
    h_new_ = w_new_ = 0;
    param_.set_scale(1.5f);
    TMatFromProto(param_, tmat_);
  }
  virtual void TearDown() {
    delete coord_;
    delete orig_;
    delete warped_;
  }

  float tmat_[9];
  TransParameter param_;
  int height_, width_, h_new_, w_new_;
  Blob<float> *coord_;
  Blob<Dtype> *orig_;
  Blob<Dtype> *warped_;

  Border border_;
  Interp interp_;

  void set_interp(const Interp &interp) {
    interp_ = interp;
  };
  void set_border(const Border &border) {
    border_ = border;
  };

  // resets parameters
  void Init(const float &scale, const float &angle, const int &height,
            const int &width) {
    height_ = height;
    width_ = width;
    param_.set_scale(scale);
    param_.set_rotation(angle);
    TMatFromProto(param_, tmat_);
    orig_->Reshape(1, 1, height_, width_);
  }

  void InitWithImage(const std::string &fname) {
    cv::Mat img = cv::imread(fname);
    if (img.channels() > 1)
      cvtColor(img, img, CV_RGB2GRAY);

    // TODO: This really assumes Dtype is float. need to check and send CV_64FC1
    // if double
    img.convertTo(img, CV_32FC1, 1. / 255);

    height_ = img.rows;
    width_ = img.cols;

    if (height_ == 0)
      LOG(FATAL) << "opencv couldn't read the image right";

    // copy image over to orig_
    orig_->Reshape(1, 1, height_, width_);
    Dtype *data = orig_->mutable_cpu_data();

    for (int i = 0; i < height_; ++i) {
      for (int j = 0; j < width_; ++j) {
        data[i * width_ + j] = img.at<float>(i, j);
      }
    }
  }

  // Set image data and coordinates for interpolating
  void SetImgAndCoord(const Dtype *img) {
    Dtype *data = orig_->mutable_cpu_data();
    for (int i = 0; i < height_ * width_; ++i) {
      data[i] = img[i];
    }
    // compute coordinate
    GenCoordMat(tmat_, height_, width_, coord_, h_new_, w_new_, border_,
                interp_);

    // set size of new img
    warped_->Reshape(1, 1, h_new_, w_new_);
  }

  void SetParam(float scale, float angle) {
    param_.set_scale(scale);
    param_.set_rotation(angle);
    TMatFromProto(param_, tmat_);

    GenCoordMat(tmat_, height_, width_, coord_, h_new_, w_new_, border_,
                interp_);
    warped_->Reshape(1, 1, h_new_, w_new_);
  }

  // Treat warped_ as top and set its diff image
  void SetTopDiffAndCoord(const Dtype *img) {
    // compute coordinate
    GenCoordMat(tmat_, height_, width_, coord_, h_new_, w_new_, border_,
                interp_);
    warped_->Reshape(1, 1, h_new_, w_new_);
    // set diff
    Dtype *diff = warped_->mutable_cpu_diff();
    for (int i = 0; i < h_new_ * w_new_; ++i) {
      diff[i] = img[i];
    }
  }

  void sub2ind(const int n, const int width, const float *coord_want_sub,
               float *coord_want_ind) {
    for (int i = 0; i < n; ++i) {
      coord_want_ind[i] =
          coord_want_sub[2 * i] * width + coord_want_sub[2 * i + 1];
    }
  }

  void printMat(const float *data, const int &row, const int &col) {
    for (int i = 0; i < row * col; ++i) {
      printf("%.3f\t", data[i]);
      if ((i + 1) % col == 0)
        printf("\n");
    }
    printf("\n");
  }
  void printMat(const double *data, const int &row, const int &col) {
    for (int i = 0; i < row * col; ++i) {
      printf("%.2f\t", data[i]);
      if ((i + 1) % col == 0)
        printf("\n");
    }
    printf("\n");
  }
};

TYPED_TEST_CASE(TransformationTest, Dtype);

// ---------- GenCoordMat stuff ----------
TYPED_TEST(TransformationTest, GenCoordMat_square) {
  this->Init(1.5f, 0, 3, 3);
  GenCoordMat(this->tmat_, this->height_, this->width_, this->coord_,
              this->h_new_, this->w_new_, this->border_, this->interp_);
  ASSERT_EQ(4, this->h_new_);
  ASSERT_EQ(4, this->w_new_);

  float coord_want_sub[16 * 2] = { 0., 0., 0., 1., 0., 1., 0., 2., 1., 0., 1.,
                                   1., 1., 1., 1., 2., 1., 0., 1., 1., 1., 1.,
                                   1., 2., 2., 0., 2., 1., 2., 1., 2., 2., };

  float coord_want[16];
  this->sub2ind(16, this->width_, coord_want_sub, coord_want);

  const float *coord_data = this->coord_->cpu_data();
  for (int i = 0; i < this->h_new_ * this->w_new_; ++i) {
    ASSERT_EQ(coord_want[i], coord_data[i]);
  }
}

TYPED_TEST(TransformationTest, GenCoordMat_square_bilinear) {
  this->set_interp(BILINEAR);
  this->Init(2.f, 0, 1, 1);
  GenCoordMat(this->tmat_, this->height_, this->width_, this->coord_,
              this->h_new_, this->w_new_, this->border_, this->interp_);
  ASSERT_EQ(2, this->h_new_);
  ASSERT_EQ(2, this->w_new_);

  float coord_want[16] = { 0., 0.,  0., 0.,  1., 1., 1.,  1.,
                           0., 0.5, 0., 0.5, 0., 0., 0.5, 0.5 };

  const float *coord_data = this->coord_->cpu_data();
  // this->printMat(coord_data, 4, 4);

  for (int i = 0; i < this->h_new_ * this->w_new_; ++i) {
    ASSERT_EQ(coord_want[i], coord_data[i]);
  }
}

TYPED_TEST(TransformationTest, GenCoordMat_rect) {
  this->Init(1.5f, 0, 4, 3);
  GenCoordMat(this->tmat_, this->height_, this->width_, this->coord_,
              this->h_new_, this->w_new_, this->border_, this->interp_);
  ASSERT_EQ(6, this->h_new_);
  ASSERT_EQ(4, this->w_new_);

  float coord_want_sub[24 * 2] = { 0., 0., 0., 1., 0., 1., 0., 2., 1., 0.,
                                   1., 1., 1., 1., 1., 2., 1., 0., 1., 1.,
                                   1., 1., 1., 2., 2., 0., 2., 1., 2., 1.,
                                   2., 2., 3., 0., 3., 1., 3., 1., 3., 2.,
                                   3., 0., 3., 1., 3., 1., 3., 2., };

  float coord_want[24];
  this->sub2ind(this->w_new_ * this->h_new_, this->width_, coord_want_sub,
                coord_want);

  const float *coord_data = this->coord_->cpu_data();
  for (int i = 0; i < this->h_new_ * this->w_new_; ++i) {
    ASSERT_EQ(coord_want[i], coord_data[i]);
  }
}

TYPED_TEST(TransformationTest, GenCoordMat_rect_rot) {
  this->set_border(REFLECT);
  this->Init(1.5f, 15, 3, 2);
  GenCoordMat(this->tmat_, this->height_, this->width_, this->coord_,
              this->h_new_, this->w_new_, this->border_);
  EXPECT_EQ(5, this->h_new_);
  EXPECT_EQ(4, this->w_new_);

  float coord_want_sub[20 * 2] = { 1., 0., 0., 1., 0., 1., 0., 0., 0., 0.,
                                   0., 0., 0., 1., 1., 0., 1., 0., 1., 0.,
                                   1., 1., 1., 1., 1., 1., 2., 0., 2., 1.,
                                   2., 1., 2., 1., 2., 0., 2., 0., 1., 1., };

  float tmat_want[9] = { 0.64395055,  -0.17254603, 0.,
                         0.17254603,  0.64395055,  0.,
                         -1.54672015, -0.62083377, 1. };
  const float *coord_data = this->coord_->cpu_data();

  for (int i = 0; i < 9; ++i) {
    ASSERT_TRUE(tmat_want[i] - this->tmat_[i] < 0.00001);
  }
  float coord_want[20];
  this->sub2ind(this->w_new_ * this->h_new_, this->width_, coord_want_sub,
                coord_want);

  for (int i = 0; i < this->h_new_ * this->w_new_; ++i) {
    ASSERT_EQ(coord_want[i], coord_data[i]);
  }
}

TYPED_TEST(TransformationTest, GenCoordMat_rect_rot2) {
  this->set_border(REFLECT);
  this->Init(.85f, -15, 5, 3);
  GenCoordMat(this->tmat_, this->height_, this->width_, this->coord_,
              this->h_new_, this->w_new_, this->border_);
  EXPECT_EQ(4, this->h_new_);
  EXPECT_EQ(3, this->w_new_);

  float coord_want_sub[12 * 2] = { 1., 1., 0., 1., 0., 2., 2., 0.,
                                   1., 1., 1., 2., 3., 0., 3., 1.,
                                   2., 2., 4., 0., 4., 1., 3., 1., };

  float tmat_want[9] = { 1.13638333,  0.30449299,  0.,
                         -0.30449299, 1.13638333,  0.,
                         -1.40008199, -1.59312282, 1. };
  const float *coord_data = this->coord_->cpu_data();

  for (int i = 0; i < 9; ++i) {
    ASSERT_TRUE(tmat_want[i] - this->tmat_[i] < 0.00001);
  }
  float coord_want[12];
  this->sub2ind(this->w_new_ * this->h_new_, this->width_, coord_want_sub,
                coord_want);

  for (int i = 0; i < this->h_new_ * this->w_new_; ++i) {
    ASSERT_EQ(coord_want[i], coord_data[i]);
  }
}

TYPED_TEST(TransformationTest, CropCenterSquare) {
  // Big to small.
  ImageSize target(6, 6);
  this->Init(1.f, 0, 11, 11);
  GenCoordMat(this->tmat_, this->height_, this->width_, this->coord_,
              this->h_new_, this->w_new_, this->border_, this->interp_);
  const float *coord_data = this->coord_->cpu_data();

  float coord_data_clipped[36];
  ImageSize original(this->width_, this->height_);
  CropCenter(coord_data, original, target, this->interp_, coord_data_clipped);
  float want[36] = { 36, 37, 38, 39, 40, 41, 47, 48, 49, 50, 51, 52,
                     58, 59, 60, 61, 62, 63, 69, 70, 71, 72, 73, 74,
                     80, 81, 82, 83, 84, 85, 91, 92, 93, 94, 95, 96 };

  for (int i = 0; i < 36; ++i) {
    ASSERT_EQ(want[i], coord_data_clipped[i]);
  }

  // Scale down.
  this->Init(1.f, 0, 3, 3);
  GenCoordMat(this->tmat_, this->height_, this->width_, this->coord_,
              this->h_new_, this->w_new_, this->border_, this->interp_);
  coord_data = this->coord_->cpu_data();
  target.width = 7;
  target.height = 7;
  original.height = this->height_;
  original.width = this->width_;
  float coord_data_clipped1[49];
  CropCenter(coord_data, original, target, this->interp_, coord_data_clipped1);

  float want1[49] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                      -1, -1, -1, 0,  1,  2,  -1, -1, -1, -1, 3,  4,  5,
                      -1, -1, -1, -1, 6,  7,  8,  -1, -1, -1, -1, -1, -1,
                      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };

  for (int i = 0; i < 49; ++i) {
    ASSERT_EQ(want1[i], coord_data_clipped1[i]);
  }

  // Scale down, uneven pad.
  this->Init(1.f, 0, 4, 4);
  GenCoordMat(this->tmat_, this->height_, this->width_, this->coord_,
              this->h_new_, this->w_new_, this->border_, this->interp_);
  coord_data = this->coord_->cpu_data();
  original.height = this->height_;
  original.width = this->width_;
  CropCenter(coord_data, original, target, this->interp_, coord_data_clipped1);

  float want2[49] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                      -1, -1, -1, 0,  1,  2,  3,  -1, -1, -1, 4,  5,  6,
                      7,  -1, -1, -1, 8,  9,  10, 11, -1, -1, -1, 12, 13,
                      14, 15, -1, -1, -1, -1, -1, -1, -1, -1, };

  for (int i = 0; i < 49; ++i) {
    ASSERT_EQ(want2[i], coord_data_clipped1[i]);
  }

  // Same size.
  this->Init(1.f, 0, 7, 7);
  GenCoordMat(this->tmat_, this->height_, this->width_, this->coord_,
              this->h_new_, this->w_new_, this->border_, this->interp_);
  coord_data = this->coord_->cpu_data();
  original.height = this->height_;
  original.width = this->width_;
  CropCenter(coord_data, original, target, this->interp_, coord_data_clipped1);

  for (int i = 0; i < 49; ++i) {
    ASSERT_EQ(i, coord_data_clipped1[i]);
  }
}

TYPED_TEST(TransformationTest, CropCenterNonSquare) {
  // Big to small.
  ImageSize target(6, 5);
  this->Init(1.f, 0, 11, 11);
  GenCoordMat(this->tmat_, this->height_, this->width_, this->coord_,
              this->h_new_, this->w_new_, this->border_, this->interp_);
  const float *coord_data = this->coord_->cpu_data();

  float cropped_coord[30];
  ImageSize original(this->width_, this->height_);
  CropCenter(coord_data, original, target, this->interp_, cropped_coord);
  float want[30] = {
    36, 37, 38, 39, 40, 41, 47, 48, 49, 50, 51, 52, 58, 59, 60,
    61, 62, 63, 69, 70, 71, 72, 73, 74, 80, 81, 82, 83, 84, 85
  };

  for (int i = 0; i < target.width * target.height; ++i) {
    ASSERT_EQ(want[i], cropped_coord[i]);
  }

  // Small to big, even padding.
  this->Init(1.f, 0, 3, 5);
  GenCoordMat(this->tmat_, this->height_, this->width_, this->coord_,
              this->h_new_, this->w_new_, this->border_, this->interp_);
  coord_data = this->coord_->cpu_data();
  target.width = 7;
  target.height = 11;
  original.height = this->height_;
  original.width = this->width_;
  float cropped_coord_1[77];
  CropCenter(coord_data, original, target, this->interp_, cropped_coord_1);

  float want1[77] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                      -1, -1, -1, 0,  1,  2,  3,  4,  -1, -1, 5,  6,  7,
                      8,  9,  -1, -1, 10, 11, 12, 13, 14, -1, -1, -1, -1,
                      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };

  for (int i = 0; i < target.width * target.height; ++i) {
    ASSERT_EQ(want1[i], cropped_coord_1[i]);
  }

  // Small to big, uneven padding.
  this->Init(1.f, 0, 3, 6);
  target.width = 9;
  target.height = 12;
  original.height = this->height_;
  original.width = this->width_;
  GenCoordMat(this->tmat_, this->height_, this->width_, this->coord_,
              this->h_new_, this->w_new_, this->border_, this->interp_);
  coord_data = this->coord_->cpu_data();
  float cropped_coord_2[108];
  CropCenter(coord_data, original, target, this->interp_, cropped_coord_2);

  float want_2[108] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                        -1, -1, -1, -1, -1, 0,  1,  2,  3,  4,  5,  -1, -1, -1,
                        6,  7,  8,  9,  10, 11, -1, -1, -1, 12, 13, 14, 15, 16,
                        17, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, };

  for (int i = 0; i < target.width * target.height; ++i) {
    ASSERT_EQ(want_2[i], cropped_coord_2[i]);
  }

  // Same size.
  this->Init(1.f, 0, 7, 3);
  target.width = 3;
  target.height = 7;
  original.height = this->height_;
  original.width = this->width_;
  GenCoordMat(this->tmat_, this->height_, this->width_, this->coord_,
              this->h_new_, this->w_new_, this->border_, this->interp_);
  coord_data = this->coord_->cpu_data();
  float cropped_coord_3[21];
  CropCenter(coord_data, original, target, this->interp_, cropped_coord_3);

  for (int i = 0; i < target.width * target.height; ++i) {
    ASSERT_EQ(i, cropped_coord_3[i]);
  }
}

TYPED_TEST(TransformationTest, CropCenterSquareBilinear) {
  this->set_interp(BILINEAR);
  // Big to small.
  ImageSize target(6, 6);
  this->Init(1.f, 0, 11, 11);
  GenCoordMat(this->tmat_, this->height_, this->width_, this->coord_,
              this->h_new_, this->w_new_, this->border_, this->interp_);
  const float *coord_data = this->coord_->cpu_data();

  // this->printMat(coord_data, 4, this->h_new_*this->w_new_);
  float coord_data_clipped[4 * 36];
  ImageSize original(this->width_, this->height_);
  // this->printMat(coord_data, 4, original.width * original.height);
  CropCenter(coord_data, original, target, this->interp_, coord_data_clipped);

  float want[36] = { 36, 37, 38, 39, 40, 41, 47, 48, 49, 50, 51, 52,
                     58, 59, 60, 61, 62, 63, 69, 70, 71, 72, 73, 74,
                     80, 81, 82, 83, 84, 85, 91, 92, 93, 94, 95, 96 };
  // printf("clipped:\n");
  // this->printMat(coord_data_clipped, 4, target_w*target_w);

  // Coefficients should be 0 because there is no interpolation.
  for (int i = 0; i < 36; ++i) {
    ASSERT_EQ(want[i], coord_data_clipped[i]);
    ASSERT_EQ(0, coord_data_clipped[i + 2 * target.width * target.width]);
    ASSERT_EQ(0, coord_data_clipped[i + 3 * target.width * target.width]);
  }
  // printf("scaling down\n");
  // scale down
  this->Init(1.f, 0, 3, 3);
  GenCoordMat(this->tmat_, this->height_, this->width_, this->coord_,
              this->h_new_, this->w_new_, this->border_, this->interp_);
  coord_data = this->coord_->cpu_data();
  // this->printMat(coord_data, 4, this->h_new_*this->w_new_);
  target.width = 7;
  target.height = 7;
  original.height = this->height_;
  original.width = this->width_;
  float coord_data_clipped1[4 * 49];
  CropCenter(coord_data, original, target, this->interp_, coord_data_clipped1);

  float want1[49] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                      -1, -1, -1, 0,  1,  2,  -1, -1, -1, -1, 3,  4,  5,
                      -1, -1, -1, -1, 6,  7,  8,  -1, -1, -1, -1, -1, -1,
                      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };

  // this->printMat(coord_data_clipped1, 4, target_w*target_w);
  // First dimension should be the same.
  for (int i = 0; i < 49; ++i) {
    ASSERT_EQ(want1[i], coord_data_clipped1[i]);
    int want_val = (coord_data_clipped1[i] == -1) ? -1 : 0;
    ASSERT_EQ(want_val,
              coord_data_clipped1[i + 2 * target.width * target.width]);
    ASSERT_EQ(want_val,
              coord_data_clipped1[i + 3 * target.width * target.width]);
  }

  // Scale down, uneven padding.
  this->Init(1.f, 0, 4, 4);
  GenCoordMat(this->tmat_, this->height_, this->width_, this->coord_,
              this->h_new_, this->w_new_, this->border_, this->interp_);
  coord_data = this->coord_->cpu_data();
  original.height = this->height_;
  original.width = this->width_;
  CropCenter(coord_data, original, target, this->interp_, coord_data_clipped1);

  float want2[49] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                      -1, -1, -1, 0,  1,  2,  3,  -1, -1, -1, 4,  5,  6,
                      7,  -1, -1, -1, 8,  9,  10, 11, -1, -1, -1, 12, 13,
                      14, 15, -1, -1, -1, -1, -1, -1, -1, -1 };

  for (int i = 0; i < 49; ++i) {
    ASSERT_EQ(want2[i], coord_data_clipped1[i]);
    int want_val = (coord_data_clipped1[i] == -1) ? -1 : 0;
    ASSERT_EQ(want_val,
              coord_data_clipped1[i + 2 * target.width * target.width]);
    ASSERT_EQ(want_val,
              coord_data_clipped1[i + 3 * target.width * target.width]);
  }

  // Same size.
  this->Init(1.f, 0, 7, 7);

  GenCoordMat(this->tmat_, this->height_, this->width_, this->coord_,
              this->h_new_, this->w_new_, this->border_, this->interp_);
  coord_data = this->coord_->cpu_data();
  original.height = this->height_;
  original.width = this->width_;
  CropCenter(coord_data, original, target, this->interp_, coord_data_clipped1);

  for (int i = 0; i < 4 * 49; ++i) {
    ASSERT_EQ(coord_data[i], coord_data_clipped1[i]);
  }
}

TYPED_TEST(TransformationTest, CropCenterNonSquareBilinear) {
  this->set_interp(BILINEAR);
  // Big to small.
  ImageSize target(6, 5);
  this->Init(1.f, 0, 11, 11);
  GenCoordMat(this->tmat_, this->height_, this->width_, this->coord_,
              this->h_new_, this->w_new_, this->border_, this->interp_);
  const float *coord_data = this->coord_->cpu_data();

  float cropped_coord[4 * 30];
  ImageSize original(this->width_, this->height_);
  CropCenter(coord_data, original, target, this->interp_, cropped_coord);
  float want[30] = {
    36, 37, 38, 39, 40, 41, 47, 48, 49, 50, 51, 52, 58, 59, 60,
    61, 62, 63, 69, 70, 71, 72, 73, 74, 80, 81, 82, 83, 84, 85
  };

  for (int i = 0; i < target.width * target.height; ++i) {
    ASSERT_EQ(want[i], cropped_coord[i]);
    ASSERT_EQ(0, cropped_coord[i + 2 * target.width * target.height]);
    ASSERT_EQ(0, cropped_coord[i + 3 * target.width * target.height]);
  }

  // Small to big, even padding.
  this->Init(1.f, 0, 3, 5);
  GenCoordMat(this->tmat_, this->height_, this->width_, this->coord_,
              this->h_new_, this->w_new_, this->border_, this->interp_);
  coord_data = this->coord_->cpu_data();
  target.width = 7;
  target.height = 11;
  original.height = this->height_;
  original.width = this->width_;
  float cropped_coord_1[4 * 77];
  CropCenter(coord_data, original, target, this->interp_, cropped_coord_1);

  float want1[77] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                      -1, -1, -1, 0,  1,  2,  3,  4,  -1, -1, 5,  6,  7,
                      8,  9,  -1, -1, 10, 11, 12, 13, 14, -1, -1, -1, -1,
                      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };

  for (int i = 0; i < target.width * target.height; ++i) {
    ASSERT_EQ(want1[i], cropped_coord_1[i]);
    int want_val = (cropped_coord_1[i] == -1) ? -1 : 0;
    ASSERT_EQ(want_val, cropped_coord_1[i + 2 * target.width * target.height]);
    ASSERT_EQ(want_val, cropped_coord_1[i + 3 * target.width * target.height]);
  }

  // Small to big, uneven padding.
  this->Init(1.f, 0, 3, 6);
  target.width = 9;
  target.height = 12;
  original.height = this->height_;
  original.width = this->width_;
  GenCoordMat(this->tmat_, this->height_, this->width_, this->coord_,
              this->h_new_, this->w_new_, this->border_, this->interp_);
  coord_data = this->coord_->cpu_data();
  float cropped_coord_2[4 * 108];
  CropCenter(coord_data, original, target, this->interp_, cropped_coord_2);

  float want_2[108] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                        -1, -1, -1, -1, -1, 0,  1,  2,  3,  4,  5,  -1, -1, -1,
                        6,  7,  8,  9,  10, 11, -1, -1, -1, 12, 13, 14, 15, 16,
                        17, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, };

  for (int i = 0; i < target.width * target.height; ++i) {
    ASSERT_EQ(want_2[i], cropped_coord_2[i]);
    int want_val = (cropped_coord_2[i] == -1) ? -1 : 0;
    ASSERT_EQ(want_val, cropped_coord_2[i + 2 * target.width * target.height]);
    ASSERT_EQ(want_val, cropped_coord_2[i + 3 * target.width * target.height]);
  }

  // Same size.
  this->Init(1.f, 0, 7, 3);
  target.width = 3;
  target.height = 7;
  original.height = this->height_;
  original.width = this->width_;
  GenCoordMat(this->tmat_, this->height_, this->width_, this->coord_,
              this->h_new_, this->w_new_, this->border_, this->interp_);
  coord_data = this->coord_->cpu_data();
  float cropped_coord_3[4 * 21];
  CropCenter(coord_data, original, target, this->interp_, cropped_coord_3);

  for (int i = 0; i < target.width * target.height; ++i) {
    ASSERT_EQ(i, cropped_coord_3[i]);
  }
}

TYPED_TEST(TransformationTest, InterpImageNN_cpu_square) {
  // 2x2 checkerboard.
  this->Init(2., 0., 2, 2);
  TypeParam img[4] = { 0, 1, 1, 0 };
  this->SetImgAndCoord(img);
  TypeParam want[16] = { 0., 0., 1., 1., 0., 0., 1., 1.,
                         1., 1., 0., 0., 1., 1., 0., 0. };

  // interpolate
  InterpImageNN_cpu(this->orig_, this->coord_->cpu_data(), this->warped_);

  const TypeParam *res = this->warped_->cpu_data();
  for (int i = 0; i < this->h_new_ * this->w_new_; ++i) {
    ASSERT_EQ(want[i], res[i]);
  }
}

TYPED_TEST(TransformationTest, InterpImageNN_cpu_square_90deg) {
  // 2x2 checkerboard.
  // this->Init(1., 90., 4, 4);
  this->Init(2., 0., 4, 4);
  // want:
  TypeParam img[16] = { 0., 0., 1., 1., 0., 0., 1., 1.,
                        1., 1., 0., 0., 1., 1., 0., 0. };
  this->SetImgAndCoord(img);
  // this->printMat(this->coord_->cpu_data(), 1, 16);
  // interpolate
  InterpImageNN_cpu(this->orig_, this->coord_->cpu_data(), this->warped_,
                    this->interp_);

  imshow(this->warped_, 1, "90 deg");

  this->set_interp(BILINEAR);
  this->Init(2., 0., 4, 4);
  this->SetImgAndCoord(img);
  InterpImageNN_cpu(this->orig_, this->coord_->cpu_data(), this->warped_,
                    this->interp_);
  imshow(this->warped_, 1, "90 deg bilinear");
  // const TypeParam* res = this->warped_->cpu_data();
  // for (int i = 0; i < this->h_new_*this->w_new_; ++i) {
  //   ASSERT_EQ(want[i], res[i]);
  // }
}

TYPED_TEST(TransformationTest, InterpImageNNBilinear_cpu_square) {
  // 2x2 checkerboard.
  this->set_interp(BILINEAR);
  this->Init(2., 0., 2, 2);
  TypeParam img[4] = { 0, 1, 1, 0 };
  this->SetImgAndCoord(img);
  // want:
  TypeParam want[16] = {
    0.000, 0.250, 0.750, 1.000, 0.250, 0.375, 0.625, 0.750,
    0.750, 0.625, 0.375, 0.250, 1.000, 0.750, 0.250, 0.000
  };

  // interpolate
  InterpImageNN_cpu(this->orig_, this->coord_->cpu_data(), this->warped_,
                    BILINEAR);
  // this->printMat(this->coord_->cpu_data(), 4, 16);
  // this->printMat(this->warped_->cpu_data(), 4, 4);

  const TypeParam *res = this->warped_->cpu_data();
  for (int i = 0; i < this->h_new_ * this->w_new_; ++i) {
    ASSERT_EQ(want[i], res[i]);
  }
}

TYPED_TEST(TransformationTest, InterpImageNNBilinear_gpu_square) {
  // 2x2 checkerboard.
  this->set_interp(BILINEAR);
  this->Init(2., 0., 2, 2);
  TypeParam img[4] = { 0, 1, 1, 0 };
  this->SetImgAndCoord(img);
  // want:
  TypeParam want[16] = {
    0.000, 0.250, 0.750, 1.000, 0.250, 0.375, 0.625, 0.750,
    0.750, 0.625, 0.375, 0.250, 1.000, 0.750, 0.250, 0.000
  };

  // interpolate
  InterpImageNN_gpu(this->orig_, this->coord_->gpu_data(), this->warped_,
                    BILINEAR);
  // this->printMat(this->coord_->cpu_data(), 4, 16);
  // this->printMat(this->warped_->cpu_data(), 4, 4);

  const TypeParam *res = this->warped_->cpu_data();
  for (int i = 0; i < this->h_new_ * this->w_new_; ++i) {
    ASSERT_EQ(want[i], res[i]);
  }
}

TYPED_TEST(TransformationTest, InterpImageNN_gpu_square) {
  // 2x2 checkerboard.
  this->Init(2., 0., 2, 2);
  TypeParam img[4] = { 0, 1, 1, 0 };
  this->SetImgAndCoord(img);
  // want:
  TypeParam want[16] = { 0., 0., 1., 1., 0., 0., 1., 1.,
                         1., 1., 0., 0., 1., 1., 0., 0. };
  // interpolate
  InterpImageNN_gpu(this->orig_, this->coord_->gpu_data(), this->warped_);

  const TypeParam *res = this->warped_->cpu_data();

  for (int i = 0; i < this->h_new_ * this->w_new_; ++i) {
    ASSERT_EQ(want[i], res[i]);
  }
}

TYPED_TEST(TransformationTest, InterpImageNN_cpu_square_rot) {
  // 2x2 checkerboard.
  this->Init(2., 90., 2, 2);
  TypeParam img[4] = { 0, 1, 1, 0 };
  this->SetImgAndCoord(img);
  // want:
  TypeParam want[16] = { 1., 1., 0., 0., 1., 1., 0., 0.,
                         0., 0., 1., 1., 0., 0., 1., 1. };
  // interpolate
  InterpImageNN_cpu(this->orig_, this->coord_->cpu_data(), this->warped_);

  const TypeParam *res = this->warped_->cpu_data();
  for (int i = 0; i < this->h_new_ * this->w_new_; ++i) {
    ASSERT_EQ(want[i], res[i]);
  }
}

TYPED_TEST(TransformationTest, InterpImageNN_gpu_square_rot) {
  // 2x2 checkerboard.
  this->Init(2., 90., 2, 2);
  TypeParam img[4] = { 0, 1, 1, 0 };
  this->SetImgAndCoord(img);
  // want:
  TypeParam want[16] = { 1., 1., 0., 0., 1., 1., 0., 0.,
                         0., 0., 1., 1., 0., 0., 1., 1. };
  // interpolate
  InterpImageNN_gpu(this->orig_, this->coord_->gpu_data(), this->warped_);

  const TypeParam *res = this->warped_->cpu_data();
  for (int i = 0; i < this->h_new_ * this->w_new_; ++i) {
    ASSERT_EQ(want[i], res[i]);
  }
}

TYPED_TEST(TransformationTest, InterpImageNN_cpu_rect_rot) {
  // 2x2 checkerboard.
  this->Init(2., 90., 4, 2);
  TypeParam img[8] = { 0, 1, 1, 0, 0, 1, 1, 0 };
  this->SetImgAndCoord(img);
  // want:
  TypeParam want[32] = { 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0.,
                         0., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0.,
                         1., 1., 0., 0., 1., 1., 0., 0., 1., 1. };
  // interpolate
  InterpImageNN_cpu(this->orig_, this->coord_->cpu_data(), this->warped_);

  const TypeParam *res = this->warped_->cpu_data();
  for (int i = 0; i < this->h_new_ * this->w_new_; ++i) {
    ASSERT_EQ(want[i], res[i]);
  }
}

TYPED_TEST(TransformationTest, PropagateError_cpu_square) {
  // 2x2 checkerboard.
  this->Init(2., 0., 2, 2);
  TypeParam diff[16] = { -1., -1., 1., 1., -1., -1., 1., 1.,
                         10., 10., 2., 2., 10., 10., 2., 2. };
  this->SetTopDiffAndCoord(diff);

  TypeParam want[4] = { -4., 4., 40., 8. };

  PropagateErrorNN_cpu(this->warped_, this->coord_->cpu_data(), this->orig_);

  const TypeParam *res = this->orig_->cpu_diff();
  for (int i = 0; i < this->height_ * this->width_; ++i) {
    ASSERT_EQ(want[i], res[i]);
  }
}

TYPED_TEST(TransformationTest, PropagateError_cpu_square_bilinear) {
  // 2x2 checkerboard.
  this->set_interp(BILINEAR);
  this->Init(2., 0., 2, 2);
  TypeParam diff[16] = { -1., -1., 1., 1., -1., -1., 1., 1.,
                         10., 10., 2., 2., 10., 10., 2., 2. };
  this->SetTopDiffAndCoord(diff);

  TypeParam want[4] = { 1.875, 4.125, 31.125, 10.875 };

  PropagateErrorNN_cpu(this->warped_, this->coord_->cpu_data(), this->orig_,
                       this->interp_);

  const TypeParam *res = this->orig_->cpu_diff();
  this->printMat(res, this->height_, this->width_);

  for (int i = 0; i < this->height_ * this->width_; ++i) {
    ASSERT_EQ(want[i], res[i]);
  }
}

TYPED_TEST(TransformationTest, PropagateError_gpu_square_bilinear) {
  // 2x2 checkerboard.
  this->set_interp(BILINEAR);
  this->Init(2., 0., 2, 2);
  TypeParam diff[16] = { -1., -1., 1., 1., -1., -1., 1., 1.,
                         10., 10., 2., 2., 10., 10., 2., 2. };
  this->SetTopDiffAndCoord(diff);

  TypeParam want[4] = { 1.875, 4.125, 31.125, 10.875 };

  PropagateErrorNN_gpu(this->warped_, this->coord_->gpu_data(), this->orig_,
                       this->interp_);

  const TypeParam *res = this->orig_->cpu_diff();
  this->printMat(res, this->height_, this->width_);

  for (int i = 0; i < this->height_ * this->width_; ++i) {
    ASSERT_EQ(want[i], res[i]);
  }
}

TYPED_TEST(TransformationTest, PropagateError_gpu_square) {
  // 2x2 checkerboard.
  this->Init(2., 0., 2, 2);
  TypeParam diff[16] = { -1., -1., 1., 1., -1., -1., 1., 1.,
                         10., 10., 2., 2., 10., 10., 2., 2. };
  this->SetTopDiffAndCoord(diff);

  TypeParam want[4] = { -4., 4., 40., 8. };

  PropagateErrorNN_gpu(this->warped_, this->coord_->gpu_data(), this->orig_);

  const TypeParam *res = this->orig_->cpu_diff();
  // printMat(res, this->height_, this->width_);
  for (int i = 0; i < this->height_ * this->width_; ++i) {
    ASSERT_EQ(want[i], res[i]);
  }
}

TYPED_TEST(TransformationTest, justchecking) {
  // 2x2 checkerboard.
  this->set_interp(BILINEAR);
  this->Init(0.5, 0., 7, 7);
  GenCoordMat(this->tmat_, this->height_, this->width_, this->coord_,
              this->h_new_, this->w_new_, this->border_, this->interp_);
  LOG(INFO) << "New size is " << this->h_new_ << " by " << this->w_new_;
  this->printMat(this->coord_->cpu_data(), 4, this->h_new_ * this->w_new_);
}

TYPED_TEST(TransformationTest, InterpImageNN_img) {
  // If this test is failing, make sure you run the test executable from the
  // base CAFFE_DIR where Makefile lives.
  this->InitWithImage(CMAKE_SOURCE_DIR "caffe/test/test_data/cat.jpg");

  imshow(this->orig_, 1, "original");
  this->SetParam(1.0f, 0.f);
  InterpImageNN_cpu(this->orig_, this->coord_->cpu_data(), this->warped_);
  imshow(this->warped_, this->param_);

  this->SetParam(0.5f, 0.f);
  InterpImageNN_cpu(this->orig_, this->coord_->cpu_data(), this->warped_);
  imshow(this->warped_, this->param_);

  this->set_border(REFLECT);
  this->SetParam(1.f, 20.f);
  InterpImageNN_cpu(this->orig_, this->coord_->cpu_data(), this->warped_);
  imshow(this->warped_, this->param_);

  this->set_border(CROP);
  this->SetParam(1.f, -20.f);
  InterpImageNN_cpu(this->orig_, this->coord_->cpu_data(), this->warped_);
  imshow(this->warped_, this->param_);

  this->SetParam(2.2f, -15.f);
  InterpImageNN_cpu(this->orig_, this->coord_->cpu_data(), this->warped_);
  imshow(this->warped_, this->param_);
}

TYPED_TEST(TransformationTest, InterpImageNN_img_bilinear) {
  this->InitWithImage(CMAKE_SOURCE_DIR "caffe/test/test_data/cat.jpg");

  this->set_interp(BILINEAR);
  imshow(this->orig_, 1, "original bilinear");
  this->SetParam(1.0f, 0.f);
  InterpImageNN_cpu(this->orig_, this->coord_->cpu_data(), this->warped_,
                    this->interp_);
  imshow(this->warped_, this->param_, 1, " bilinear");

  this->SetParam(0.5f, 0.f);
  InterpImageNN_cpu(this->orig_, this->coord_->cpu_data(), this->warped_,
                    this->interp_);
  imshow(this->warped_, this->param_, 1, " bilinear");

  this->set_border(REFLECT);
  this->SetParam(1.f, 20.f);
  InterpImageNN_cpu(this->orig_, this->coord_->cpu_data(), this->warped_,
                    this->interp_);
  imshow(this->warped_, this->param_, 1, " bilinear");

  this->set_border(CROP);
  this->SetParam(1.f, -20.f);
  InterpImageNN_cpu(this->orig_, this->coord_->cpu_data(), this->warped_,
                    this->interp_);
  imshow(this->warped_, this->param_, 1, " bilinear");

  this->SetParam(2.2f, -15.f);
  InterpImageNN_cpu(this->orig_, this->coord_->cpu_data(), this->warped_,
                    this->interp_);
  imshow(this->warped_, this->param_, 1, " bilinear");
}

TYPED_TEST(TransformationTest, InterpImageNN_img_bilinear_gpu) {
  this->InitWithImage(CMAKE_SOURCE_DIR "caffe/test/test_data/cat.jpg");

  this->set_interp(BILINEAR);
  imshow(this->orig_, 1, "original bilinear gpu");
  this->SetParam(1.0f, 0.f);
  InterpImageNN_gpu(this->orig_, this->coord_->gpu_data(), this->warped_,
                    this->interp_);
  imshow(this->warped_, this->param_, 1, " bilinear gpu");

  this->SetParam(0.5f, 0.f);
  InterpImageNN_gpu(this->orig_, this->coord_->gpu_data(), this->warped_,
                    this->interp_);
  imshow(this->warped_, this->param_, 1, " bilinear gpu");

  this->set_border(REFLECT);
  this->SetParam(1.f, 20.f);
  InterpImageNN_gpu(this->orig_, this->coord_->gpu_data(), this->warped_,
                    this->interp_);
  imshow(this->warped_, this->param_, 1, " bilinear gpu");

  this->set_border(CROP);
  this->SetParam(1.f, -20.f);
  InterpImageNN_gpu(this->orig_, this->coord_->gpu_data(), this->warped_,
                    this->interp_);
  imshow(this->warped_, this->param_, 1, " bilinear gpu");

  this->SetParam(2.2f, -15.f);
  InterpImageNN_gpu(this->orig_, this->coord_->gpu_data(), this->warped_,
                    this->interp_);
  imshow(this->warped_, this->param_, 1, " bilinear gpu");
}

//----------GenCoordMat direct one ----------
TYPED_TEST(TransformationTest, GenCoordMat) {
  this->InitWithImage(CMAKE_SOURCE_DIR "caffe/test/test_data/cat.jpg");

  Blob<float> *coord = new Blob<float>(false);
  float scale = 1.2;
  float rotation = 20.;
  GenCoordMatCrop(scale, rotation, this->height_, this->width_, coord);

  this->warped_->Reshape(1, 1, this->height_, this->width_);
  InterpImageNN_cpu(this->orig_, coord->cpu_data(), this->warped_);

  // imshow(this->warped_, 1, "direct coord");
  // cv::waitKey(1);

  delete coord;
}

//----------GenCoordMat direct one ----------
TYPED_TEST(TransformationTest, GenCoordMat_bilinear) {
  this->set_interp(BILINEAR);
  this->InitWithImage(CMAKE_SOURCE_DIR "caffe/test/test_data/cat.jpg");

  Blob<float> *coord = new Blob<float>(false);
  float scale = 1.2;
  float rotation = 20.;
  GenCoordMatCrop(scale, rotation, this->height_, this->width_, coord,
                  this->border_, this->interp_);

  this->warped_->Reshape(1, 1, this->height_, this->width_);
  InterpImageNN_cpu(this->orig_, coord->cpu_data(), this->warped_,
                    this->interp_);

  // imshow(this->warped_, 1, "direct coord bilinear");
  // cv::waitKey(1);

  delete coord;
}

//----------Count Switches ----------
TEST(CountSwitches, CountSwitches) {
  int N = 20, num_t = 6;

  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(0, num_t - 1);

  // Copy host_vector to raw array (bc i want to test this with syncedmem)
  SyncedMemory mem(N * sizeof(float));
  float *cpu_data = reinterpret_cast<float *>(mem.mutable_cpu_data());

  std::vector<int> want;
  want.resize(num_t);
  // Fill with rand values, as we count
  int tmp;
  for (int i = 0; i < N; ++i) {
    tmp = dist(rng);
    // printf("%d ", tmp);
    cpu_data[i] = static_cast<float>(tmp);
    ++want[tmp];
  }
  // printf("\n");

  float *gpu_data = reinterpret_cast<float *>(mem.mutable_gpu_data());

  int counter[num_t];
  memset(counter, 0, sizeof(int) * num_t);
  CountSwitches(gpu_data, N, num_t, counter);

  for (int i = 0; i < num_t; ++i) {
    // printf("(%d %d) ", i, counter[i]);
    ASSERT_EQ(want[i], counter[i]);
  }
  // printf("\n");
}
}
