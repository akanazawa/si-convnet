// Angjoo Kanazawa 2013

#ifndef CAFFE_UTIL_TRANSF_H_
#define CAFFE_UTIL_TRANSF_H_

#include <limits>

#include "caffe/blob.hpp"

namespace caffe {

const float PI_F = 3.14159265358979f;

enum Direction { RIGHT, LEFT };
// CROP is zero-padding, CLAMP is border replicate, REFLECT is mirror.
enum Border { CROP, CLAMP, REFLECT };
enum Interp { NN, BILINEAR };

struct ImageSize {
  ImageSize() : width(0), height(0) {};
  ImageSize(int width, int height) : width(width), height(height) {};
  int width;
  int height;
};

// AJ: Function to make the 3x3 matrices from proto
void TMatFromProto(const TransParameter &param, float *tmat,
                   bool invert = false);

// shouldn't be used: (doens't preserve correspondence at convolution center)
void TMatToCanonical(const TransParameter &param, const int &cano_height,
                     const int &height, float *tmat);
// Combines transformation matrices. If right is true, the new transformation
// matrix is
// multiplied to the existing one from the right.
void AddRotation(const float &angle, float *mat, const Direction dir = RIGHT);
void AddScale(const float &scale, float *mat, const Direction dir = RIGHT);
void AddTranslation(const float &dx, const float &dy, float *mat,
                    const Direction dir = RIGHT);
// m = m*t
void AddTransform(float *mat, const float *tmp, const Direction dir = RIGHT);

void GetNewSize(const int &height, const int &width, const float *mat,
                int &height_new, int &width_new);

void Invert3x3(float *A);
// Generates coordinate matrix for backward mapping .
void GenCoordMat(float *tmat, const int &height, const int &width,
                 Blob<float> *coord, int &height_new, int &width_new,
                 const Border &border = CROP, const Interp &interp = NN);

void generate_nn_coord(const int &height, const int &width,
                       const int &height_new, const int &width_new,
                       const Border &border, const float *coord_data_res,
                       float *&coord_data);

void generate_bilinear_coord(const int &height, const int &width,
                             const int &height_new, const int &width_new,
                             const Border &border, const float *coord_data_res,
                             float *&coord_data);

// This one doesn't change the size. Used in jittering the data with scale/rotation.
void GenCoordMatCrop(const float &scale, const float &rotation,
                     const int &height, const int &width, Blob<float> *coord,
                     const Border &border = CROP, const Interp &interp = NN);

// Generates identity coordinates.
void GenBasicCoordMat(float *coord, const int &width, const int &height);
// identity coordinates in indices
void GenBasicCoordMatInds(const int &width, const int &height,
                          Blob<float> *coord);

template <typename Dtype> void Reflect(Dtype &val, const int size);
template <typename Dtype> void Clamp(Dtype &val, const int size);

// Crop the coordinates at center to go back to canonical shape
void CropCenter(const float *coord, const ImageSize &original,
                const ImageSize &target, const Interp &interp,
                float *coord_new);

template <typename Dtype>
void InterpImageNN_cpu(const Blob<Dtype> *orig, const float *coord,
                       Blob<Dtype> *warped, const Interp &interp = NN);
template <typename Dtype>
void nn_interpolation(const Blob<Dtype> *&orig, const float *&coord,
                      Blob<Dtype> *&warped);
template <typename Dtype>
void bilinear_interpolation(const Blob<Dtype> *&orig, const float *&coord,
                            Blob<Dtype> *&warped);

template <typename Dtype>
void PropagateErrorNN_cpu(const Blob<Dtype> *top, const float *coord,
                          Blob<Dtype> *bottom, const Interp &interp = NN);
template <typename Dtype>
void nn_propagation(const Blob<Dtype> *&top, const float *&coord,
                    Blob<Dtype> *&bottom);
template <typename Dtype>
void bilinear_propagation(const Blob<Dtype> *&top, const float *&coord,
                          Blob<Dtype> *&bottom);

// gpu functions
template <typename Dtype>
void InterpImageNN_gpu(const Blob<Dtype> *orig, const float *coord,
                       Blob<Dtype> *warped, const Interp &interp = NN);

template <typename Dtype>
void PropagateErrorNN_gpu(const Blob<Dtype> *top, const float *coord,
                          Blob<Dtype> *bottom, const Interp &interp = NN);

template <typename Dtype>
void MaxTransSetSwitch_gpu(const Dtype *A, Dtype *B, int count,
                           const float *coord, int sheet_count, float *switchD,
                           int tIndex);

template <typename Dtype>
void ErrorPropagateDownpoolNN_gpu_single(
    const Dtype *top, const int &t_id, const int &topCount,
    const int &topSheetCount, const float *switchD, const float *coord,
    Dtype *bottom, const int &bottomSheetCount, const int &W,
    const Interp &interp = NN);

// a function that uses thrust to count the usage of each transformation.
// switch_data is device memory, counter is host memory
void CountSwitches(float *switch_data, int n, int num_t, int *counter);

} // namespace caffe

#endif // CAFFE_UTIL_TRANSF_H_
