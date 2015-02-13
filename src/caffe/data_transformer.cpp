#include <string>

#include "caffe/data_transformer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/transformation.hpp"

namespace caffe {

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const int batch_item_id,
                                       const Datum& datum,
                                       const Dtype* mean,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int channels = datum.channels();
  const int height = datum.height();
  const int width = datum.width();
  const int size = datum.channels() * datum.height() * datum.width();

  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const Dtype scale = param_.scale();
  const bool do_preproc = param_.has_preproc();

  if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
               << "set at the same time.";
  }

  // AJ: for preprocessing (spatial scale/rotation)
  Blob<float> *coord = new Blob<float>(false);
  const float *coord_data = NULL;
  // If no preprocessing, this is just fixed
  Border border;
  PreprocParameter preproc;
  if (!do_preproc) {
    crop_size ? GenBasicCoordMatInds(crop_size, crop_size, coord)
      : GenBasicCoordMatInds(width, height, coord);
    coord_data = coord->cpu_data();
  } else {
    preproc = param_.preproc();
    border = static_cast<Border>(preproc.border());
  }
  // AJ:PREPROCESS
  if (do_preproc) {
    // generate random scale & rot from the range
    float prep_scale =
        preproc.has_min_scale()
            ? (static_cast<float>(std::rand()) / RAND_MAX) *
                      (preproc.max_scale() - preproc.min_scale()) +
                  preproc.min_scale()
            : 1.;
    float rot =
        preproc.has_min_rot()
            ? (std::rand() % (int)(preproc.max_rot() - preproc.min_rot())) +
                  preproc.min_rot()
            : 0.;
    DLOG(INFO) << "using scale " << prep_scale << " rotation " << rot;
    crop_size
        ? GenCoordMatCrop(prep_scale, rot, crop_size, crop_size, coord, border)
        : GenCoordMatCrop(prep_scale, rot, height, width, coord, border);
    coord_data = coord->cpu_data();
  }

  if (crop_size) {
    CHECK(data.size()) << "Image cropping only support uint8 data";
    int h_off, w_off;
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
      h_off = Rand() % (height - crop_size);
      w_off = Rand() % (width - crop_size);
    } else {
      h_off = (height - crop_size) / 2;
      w_off = (width - crop_size) / 2;
    }
    if (mirror && Rand() % 2) {
      // Copy mirrored version
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            const int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + (crop_size - 1 - w);
	    const int data_index_original =
	      (c * height + h + h_off) * width + w + w_off;
            const int ind_warped = h * crop_size + w;
            const int ind_orig = static_cast<int>(coord_data[ind_warped]);
            if (ind_orig < 0) {
              transformed_data[top_index] = (0 - mean[data_index_original]) * scale;
            } else {
              const int h_orig = ind_orig / crop_size;
              const int w_orig = ind_orig % crop_size;
              const int data_index =
                  (c * height + h_orig + h_off) * width + w_orig + w_off;
              const Dtype datum_element =
                  static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
              transformed_data[top_index] =
                  (datum_element - mean[data_index_original]) * scale;
            }
          }
        }
      }
    } else {
      // Normal copy
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            const int top_index = ((batch_item_id * channels + c) * crop_size + h)
	      * crop_size + w;
	    const int data_index_original =
	      (c * height + h + h_off) * width + w + w_off;
	    const int ind_warped = h * crop_size + w;
	    const int ind_orig = static_cast<int>(coord_data[ind_warped]);
	    if (ind_orig < 0) {
              transformed_data[top_index] = (0 - mean[data_index_original]) * scale;
	    } else {
	      const int h_orig = ind_orig / crop_size;
	      const int w_orig = ind_orig % crop_size;
              const int data_index =
		(c * height + h_orig + h_off) * width + w_orig + w_off;
	      const Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
	      transformed_data[top_index] =
                (datum_element - mean[data_index_original]) * scale;
	    }
          }
        }
      }
    }
  } else {
    // we will prefer to use data() first, and then try float_data()
    if (data.size()) {
      for (int j = 0; j < size; ++j) {
	// // w = j % width;
	// // h = ( j / width ) % height;
	// // c = ( j / width ) / height;
	const int ind_warped = j % (height * width);
	// CHECK_EQ(h*width + w, ind_warped) << "h: " << h << " w: " << w
	// 				    << " c: " << c << " j " << j;
	int ind_orig = static_cast<int>(coord_data[ind_warped]);
	if (ind_orig < 0) {
	  transformed_data[j + batch_item_id * size] =
	    (static_cast<Dtype>(0.) - mean[j]) * scale;
	} else {
	  ind_orig += (j - ind_warped);
	  Dtype datum_element = static_cast<Dtype>((uint8_t)data[ind_orig]);
	  transformed_data[j + batch_item_id * size] =
            (datum_element - mean[j]) * scale;
	}
      }
    } else {
      for (int j = 0; j < size; ++j) {
	const int ind_warped = j % (height * width);
	int ind_orig = static_cast<int>(coord_data[ind_warped]);
	if (ind_orig < 0) {
	  transformed_data[j + batch_item_id * size] =
	    (static_cast<Dtype>(0.) - mean[j]) * scale;
	} else {
	  ind_orig += (j - ind_warped);
	  transformed_data[j + batch_item_id * size] =
	    (datum.float_data(ind_orig) - mean[j]) * scale;
	}
      }
    }
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = (phase_ == Caffe::TRAIN) &&
      (param_.mirror() || param_.crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
unsigned int DataTransformer<Dtype>::Rand() {
  CHECK(rng_);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return (*rng)();
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
