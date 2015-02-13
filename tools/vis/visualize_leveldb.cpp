// Angjoo Kanazawa
// show content of leveldb as caffe sees it
// Usage:
//    visualize_leveldb input_leveldb_file

#include <iostream>
#include <fstream>

#include <glog/logging.h>
#include <leveldb/db.h>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/util/imshow.hpp"
#include "caffe/proto/caffe.pb.h"

void visualize_leveldb(const char *db_filename) {
  std::ifstream db_file(db_filename, std::ios::binary);
  CHECK(db_file) << "leveldb file doesn't exist " << db_filename;

  // open leveldb
  leveldb::DB *db_orig;
  leveldb::Options options;
  options.create_if_missing = false;
  options.error_if_exists = false;
  leveldb::Status status_orig =
      leveldb::DB::Open(options, db_filename, &db_orig);
  CHECK(status_orig.ok()) << "Failed to open leveldb " << db_filename;

  // get total number of elements in this db (can't seem to find that function)
  std::vector<std::string> keys;
  int counter = 0;
  leveldb::Iterator *it = db_orig->NewIterator(leveldb::ReadOptions());
  CHECK(it->status().ok()); // Check for any errors found during the scan

  // get the size
  caffe::Datum datum;
  char window_name[100];
  it->SeekToFirst();
  datum.ParseFromString(it->value().ToString());
  caffe::Blob<float> *img = new caffe::Blob<float>(
      1, datum.channels(), datum.height(), datum.width());
  int size = datum.channels() * datum.height() * datum.width();
  float *img_data = img->mutable_cpu_data();
  const std::string &data = datum.data();
  // for norb
  caffe::Blob<float> *norb;
  float *norb_data;
  if (datum.channels() == 2) {
    norb = new caffe::Blob<float>(2, 1, datum.height(), datum.width());
    norb_data = norb->mutable_cpu_data();
  }
  LOG(INFO) << "Channels: " << datum.channels();

  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    datum.ParseFromString(it->value().ToString());
    if (data.size()) {
      for (int j = 0; j < size; ++j) {
        img_data[j] = static_cast<float>((uint8_t)data[j]);
      }
    } else {
      for (int j = 0; j < size; ++j) {
        img_data[j] = datum.float_data(j);
      }
    }
    if (datum.has_label() && datum.label() != 40)
      snprintf(window_name, 100, "label %d", datum.label());

    if (datum.channels() == 2) { // for norb
      std::copy(img_data, img_data + datum.height() * datum.width(), norb_data);
      std::copy(img_data + datum.height() * datum.width(),
                img_data + 2 * datum.height() * datum.width(),
                norb_data + datum.height() * datum.width());
      caffe::montage(norb, std::string(window_name));
    } else {
      caffe::imshow(img, 1, std::string(window_name));
    }

    cv::waitKey(2);
    // LOG_EVERY_N(INFO, 500) << "image " << counter;
    ++counter;
  }
  CHECK(it->status().ok()); // Check for any errors found during the scan

  LOG(INFO) << "Total of " << counter << " items found.";

  // clean up
  delete img;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage:\n"
           "   visualize_leveldb input_leveldb_file\n");
  } else {
    google::InitGoogleLogging(argv[0]);
    visualize_leveldb(argv[1]);
  }
  return 0;
}
