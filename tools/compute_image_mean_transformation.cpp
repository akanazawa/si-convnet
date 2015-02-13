// Copyright 2013 Yangqing Jia
// Angjoo: This one can transform the input. Rotation is in degrees.
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

int main(int argc, char **argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc != 7) {
    LOG(ERROR)
        << "Usage: compute_image_mean_transformation input_leveldb output_file"
        << " min_sc max_sc min_rot max_rot";
    return (0);
  }

  float min_sc = atof(argv[3]);
  float max_sc = atof(argv[4]);
  float min_rot = atof(argv[5]);
  float max_rot = atof(argv[6]);

  // Open leveldb
  leveldb::DB *db;
  leveldb::Options options;
  options.create_if_missing = false;

  LOG(INFO) << "Opening leveldb " << argv[1];
  leveldb::Status status = leveldb::DB::Open(options, argv[1], &db);
  CHECK(status.ok()) << "Failed to open leveldb " << argv[1];

  leveldb::ReadOptions read_options;
  read_options.fill_cache = false;
  leveldb::Iterator *it = db->NewIterator(read_options);
  it->SeekToFirst();

  // set size info
  Datum datum;
  BlobProto sum_blob;
  int count = 0;
  datum.ParseFromString(it->value().ToString());
  sum_blob.set_num(1);
  sum_blob.set_channels(datum.channels());
  sum_blob.set_height(datum.height());
  sum_blob.set_width(datum.width());
  const int data_size = datum.channels() * datum.height() * datum.width();
  for (int i = 0; i < datum.data().size(); ++i) {
    sum_blob.add_data(0.);
  }
  caffe::Blob<float> *coord;
  const float *coord_data;
  // start collecting
  LOG(INFO) << "Starting Iteration";
  uint8_t max_val = 0;
  uint8_t min_val = 255;
  double mean = 0;
  uint8_t val = 0;
  int ind_warped, ind_orig;
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    // load image
    datum.ParseFromString(it->value().ToString());
    const string& data = datum.data();
    CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
                                     << data.size();

    // Pick scale/rot:
    float prep_scale =
        (static_cast<float>(std::rand()) / RAND_MAX) * (max_sc - min_sc) +
        min_sc;
    float rot = (min_rot == max_rot) && min_rot == 0
                    ? 0
                    : (std::rand() % (int)(max_rot - min_rot)) + min_rot;

    // get transformed coordinates
    caffe::GenCoordMatCrop(prep_scale, rot, datum.height(), datum.width(),
                           coord);
    coord_data = coord->cpu_data();
    double mean_t = 0.;
    for (int i = 0; i < data.size(); ++i) {
      // transform image:
      ind_warped = i % (datum.height() * datum.width());
      ind_orig = coord_data[ind_warped];
      if (ind_orig >= 0) {
        val = (uint8_t)data[ind_orig];
      } else {
        val = 0;
      }
      sum_blob.set_data(i, sum_blob.data(i) + val);
      if (val > max_val) {
        max_val = val;
      } else if (val < min_val) {
        min_val = val;
      }
      mean_t += double(val);
    }

    // caffe::Blob<float> vis_here;
    // vis_here.FromProto(sum_blob);
    // caffe::imshow(&vis_here, 1, "mean img");
    // cv::waitKey(0);

    mean += mean_t / data.size();
    ++count;
    if (count % 10000 == 0) {
      LOG(ERROR) << "Processed " << count << " files.";
    }
  }
  LOG(INFO) << "min/max val [" << int(min_val) << ", " << int(max_val)
            << "] mean val " << mean / count;

  for (int i = 0; i < sum_blob.data_size(); ++i) {
    sum_blob.set_data(i, sum_blob.data(i) / count);
  }

  caffe::Blob<float> vis;
  vis.FromProto(sum_blob);
  caffe::imshow(&vis, 1, "mean img");
  cv::waitKey(0);

  google::protobuf::RepeatedField<float> *tmp = sum_blob.mutable_data();
  std::vector<float> mean_data(tmp->begin(), tmp->end());
  double sum = std::accumulate(mean_data.begin(), mean_data.end(), 0.0);
  double mean2 = sum / mean_data.size();
  double sq_sum = std::inner_product(mean_data.begin(), mean_data.end(),
                                     mean_data.begin(), 0.0);
  double stdev = std::sqrt(sq_sum / mean_data.size() - mean2 * mean2);

  LOG(INFO) << "mean of mean image: " << mean2 << " std: " << stdev;

  // Write to disk
  char buff[200];
  if ((min_rot == max_rot) && min_rot == 0) {
    snprintf(buff, 200, "_sc_%.1f_%.1f", min_sc, max_sc);
  } else if ((min_sc == max_sc) && min_sc == 1) {
    snprintf(buff, 200, "_rot_%.1f_%.1f", min_rot, max_rot);
  } else {
    snprintf(buff, 200, "_sc_%.1f_%.1f_rot_%.1f_%.1f", min_sc, max_sc, min_rot,
             max_rot);
  }
  std::string strbuff(buff);
  std::string name(argv[2]);
  name += strbuff;
  char final[2000];
  strncpy(final, name.c_str(), sizeof(final));
  final[sizeof(final) - 1] = 0;
  LOG(INFO) << "Write to " << final;

  WriteProtoToBinaryFile(sum_blob, final);

  delete db;
  delete coord;
  return 0;
}
