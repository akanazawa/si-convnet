// Angjoo: This script applies random transforms specified by the input to all images in a leveldb and saves it to a new one.
// Rotation is in degrees.
// Usage: compute_transformed_leveldb input_leveldb output_leveldb min_sc max_sc min_rot max_rot
#include <glog/logging.h>
#include <leveldb/db.h>
#include <stdint.h>

#include <string>
#include <numeric>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <google/protobuf/repeated_field.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/imshow.hpp"
#include "caffe/util/transformation.hpp"

using caffe::Datum;
using caffe::BlobProto;
using std::string;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 7) {
    LOG(ERROR) << "Usage: compute_transformed_leveldb input_leveldb " <<
      "output_leveldb min_sc max_sc min_rot max_rot (interpolation[0=NN,1=BILINEAR])";
    return(0);
  }
  // set random seed
  std::srand(1216);  
  float min_sc = atof(argv[3]);
  float max_sc = atof(argv[4]);
  CHECK( min_sc >= 0 && min_sc <= max_sc );
  float min_rot = atof(argv[5]);  
  float max_rot = atof(argv[6]);  
  CHECK( min_rot <= 360 && max_rot <= 360 );

  caffe::Interp interp = (argc == 8) ? static_cast<caffe::Interp>(atoi(argv[7])) : caffe::BILINEAR;

  // Open input leveldb
  leveldb::DB* db_in;
  leveldb::Options options;
  options.create_if_missing = false;

  LOG(INFO) << "Opening input leveldb " << argv[1];
  leveldb::Status status = leveldb::DB::Open(options, argv[1], &db_in);
  CHECK(status.ok()) << "Failed to open leveldb " << argv[1];
  leveldb::ReadOptions read_options;
  read_options.fill_cache = false;
  leveldb::Iterator* it = db_in->NewIterator(read_options);
  it->SeekToFirst();

  // set size info
  Datum datum;
  BlobProto transformed_blob;
  int count = 0;
  datum.ParseFromString(it->value().ToString());
  CHECK(it->status().ok());  // Check for any errors found during the scan
  int channels = datum.channels();
  int height = datum.height();
  int width = datum.width();
  const int data_size = datum.channels() * datum.height() * datum.width();
  // transformed image have the same size
  Datum datum_transformed;
  datum_transformed.set_channels(channels);
  datum_transformed.set_height(height);
  datum_transformed.set_width(width);

  // Open output leveldb
  leveldb::DB* db_out;
  options.create_if_missing = true;
  options.error_if_exists = true;
  LOG(INFO) << "Opening output leveldb " << argv[2];
  leveldb::Status status_out = leveldb::DB::Open(options, argv[2], &db_out);
  CHECK(status_out.ok()) << "Failed to open leveldb " << argv[2]
		     << ". Is it already existing?";

  const int kMaxKeyLength = 10;
  char key[kMaxKeyLength];
  std::string value;
  caffe::TransParameter tparam;

  // transformation coordinates
  caffe::Blob<float>* original = new caffe::Blob<float>(1, channels, height, width, false);
  caffe::Blob<float>* transformed = new caffe::Blob<float>(1, channels, height, width, false);
  caffe::Blob<float>* coord = new caffe::Blob<float>(false);
  float tmat[9];

  float* img_data;
  // start collecting
  LOG(INFO) << "Starting Iteration";
  int size_in_datum;
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    // load image
    datum.ParseFromString(it->value().ToString());
    const string& data = datum.data();
    CHECK(data.size()) << "Image cropping only support uint8 data";
    CHECK_EQ(data.size(), data_size) << "Incorrect data field size " << data.size();

    img_data = original->mutable_cpu_data();
    // copy data first
    for (int j = 0; j < data_size; ++j ) {
      img_data[j] = static_cast<float>((uint8_t)data[j]);
    }

    // Pick scale/rot:
    float prep_scale = (static_cast<float>(std::rand()) / RAND_MAX)
                       * (max_sc - min_sc) + min_sc;
    float rot = (min_rot == max_rot) && min_rot == 0 ? 0 :
      (std::rand() % (int)(max_rot - min_rot)) + min_rot;

    // get transformed coordinates
    caffe::GenCoordMatCrop(prep_scale, rot, height, width, coord, caffe::CROP, interp); 
    // caffe::InterpImageNN_cpu(original, coord->cpu_data(), transformed, interp);
    caffe::InterpImageNN_gpu(original, coord->gpu_data(), transformed, interp);
    // caffe::imshow(original, 1, "before");    
    // tparam.set_scale(prep_scale);
    // tparam.set_rotation(rot);
    // caffe::imshow(transformed, tparam, 1, "after");
    // caffe::imshow(transformed, 1, "after");
    // cv::waitKey(2);

    // Turn transformed into Blob then to datum
    transformed->ToProto(&transformed_blob);
    datum_transformed.clear_data();
    datum_transformed.clear_float_data();
    string* datum_t_string = datum_transformed.mutable_data();
    for (int i = 0; i < data.size(); ++i) {
      datum_t_string->push_back(static_cast<char>(transformed_blob.data(i)));
    }
    datum_transformed.set_label(datum.label());    
    
    // finally save the datum
    datum_transformed.SerializeToString(&value);
    snprintf(key, kMaxKeyLength, "%08d", count);
    db_out->Put(leveldb::WriteOptions(), std::string(key), value);

    ++count;
    if (count % 10000 == 0) {
      LOG(ERROR) << "Processed " << count << " files.";
    }
  }

  delete db_in;
  delete db_out;
  delete coord;
  delete original;
  delete transformed;
  return 0;
}
