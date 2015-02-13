// Angjoo
//
// This script converts the NORB dataset to the leveldb format used
// by caffe to perform classification.
// Usage:
//    convert_norb_data input_image_file input_label_file output_db_file
// The MNIST dataset could be downloaded at
//    http://yann.lecun.com/exdb/mnist/
// Assumes little endian machine

#include <google/protobuf/text_format.h>
#include <glog/logging.h>
#include <leveldb/db.h>

#include <stdint.h>
#include <fstream> // NOLINT(readability/streams)
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/proto/caffe.pb.h"
enum Magic {
  SINGLE = 0x1E3D4C51,  // single precision 32
  PACKED = 0X1E3D4C52,  // PACKED MATRIX ??
  DOUBLE = 0X1E3D4C53,  // DOUBLE PRECISION 64
  INTEGER = 0X1E3D4C54, // INTEGER MATRIX 32
  BYTE = 0X1E3D4C55,    // BYTE MATRIX 8
  SHORT = 0X1E3D4C56,   // SHORT MATRIX 16
};

using std::string;

const string kNorbTrainName =
    "/norb-5x46789x9x18x6x2x108x108-training-%02d-%s.mat";
const string kNorbTestName =
    "/norb-5x01235x9x18x6x2x108x108-testing-%02d-%s.mat";

const string kTrainFilename = "/norb-train-leveldb";
const string kTestFilename = "/norb-test-leveldb";

const int kNorbImagesNBytes = 23328; // 2*108*108;
const int kNorbBatchSize = 29160;
const int kNorbTrainBatches = 2;
const int kNorbTestBatches = 2;
const int kNorbResize = 48; // resize images to 48 x 48
const int kMaxKeyLength = 30;

void read_header(std::ifstream *file, uint32_t &num, uint32_t &channels,
                 uint32_t &height, uint32_t &width) {
  // Read the magic and the meta data
  uint32_t magic;
  uint32_t ndim;
  file->read(reinterpret_cast<char *>(&magic), 4);
  file->read(reinterpret_cast<char *>(&ndim), 4);
  // LOG(INFO) << "magic: " << Magic(magic) << "ndim: " << ndim;;

  file->read(reinterpret_cast<char *>(&num), 4);
  file->read(reinterpret_cast<char *>(&channels), 4);
  file->read(reinterpret_cast<char *>(&height), 4);
  if (ndim == 4)
    file->read(reinterpret_cast<char *>(&width), 4);
}

int prepare_norb(std::ifstream *image_file, std::ifstream *label_file,
                 caffe::Datum *datum) {
  // read header
  uint32_t label_num, num, channels, height, width;
  read_header(label_file, label_num, channels, height, width);
  read_header(image_file, num, channels, height, width);

  LOG(INFO) << " A total of " << num << " items "
            << "Heigh: " << height << " Width: " << width
            << " Channel: " << channels;

  CHECK((label_num == num) & (num == kNorbBatchSize))
      << "num items in label " << label_num << " and data " << num << " and "
      << kNorbBatchSize << " must be the same";

  CHECK(kNorbImagesNBytes == channels * height * width)
      << " not reading the image dimensions correctly!!";

  datum->set_channels(channels);
  datum->set_height(height);
  datum->set_width(width);

  return num;
}

void mat_to_datum(const cv::Mat im0, const cv::Mat im1, caffe::Datum *datum) {
  datum->set_channels(2);
  datum->set_height(im0.rows);
  datum->set_width(im0.cols);
  datum->clear_data();
  datum->clear_float_data();
  string *datum_string = datum->mutable_data();
  for (int c = 0; c < 2; ++c) {
    for (int h = 0; h < im0.rows; ++h) {
      for (int w = 0; w < im0.cols; ++w) {
        (c == 0) ? datum_string->push_back(im0.at<char>(h, w))
                 : datum_string->push_back(im1.at<char>(h, w));
        // datum_string->push_back(static_cast<char>(im0.at<cv::Vec3b>(h,
        // w)[0])) :
        // datum_string->push_back(static_cast<char>(im1.at<cv::Vec3b>(h,
        // w)[0])) ;
      }
    }
  }
}

void store_norb_batches(const string &input_folder, const string &dataset_name,
                        int batch_size, leveldb::DB *db) {

  char key[kMaxKeyLength];
  char fname_buffer[300];
  string value;
  caffe::Datum datum;
  caffe::Datum datum_sm;
  uint32_t label; // label are ints
  char *pixels = new char[kNorbImagesNBytes];

  // batch counts from 1
  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    // Open data file
    snprintf(fname_buffer, 300, dataset_name.c_str(), batch_id + 1, "dat");
    string fname = input_folder + fname_buffer;
    std::ifstream image_file(fname.c_str(), std::ios::in | std::ios::binary);
    CHECK(image_file) << "Unable to open file " << fname;
    // Open label file
    snprintf(fname_buffer, 300, dataset_name.c_str(), batch_id + 1, "cat");
    fname = input_folder + fname_buffer;
    std::ifstream label_file(fname.c_str(), std::ios::in | std::ios::binary);
    CHECK(image_file) << "Unable to open file " << fname;
    // skip the headers..
    int num = prepare_norb(&image_file, &label_file, &datum);
    for (int itemid = 0; itemid < num; ++itemid) {
      // LOG_EVERY_N(INFO, 500) << "batch " << batch_id << " reading image " <<
      // itemid << " label: " << label;
      image_file.read(pixels, kNorbImagesNBytes);
      label_file.read(reinterpret_cast<char *>(&label), 4);
      // RESIZE
      cv::Mat im0(datum.height(), datum.width(), CV_8UC1, pixels, 0);
      cv::Mat im1(datum.height(), datum.width(), CV_8UC1,
                  pixels + datum.height() * datum.width(), 0);
      // cv::namedWindow("c1", CV_WINDOW_NORMAL );// Create a window for
      // display.
      // cv::namedWindow("c2", CV_WINDOW_NORMAL );// Create a window for
      // display.
      // cv::imshow("c1", im0);
      // cv::imshow("c2", im1);
      cv::resize(im0, im0, cv::Size(kNorbResize, kNorbResize), 0, 0,
                 cv::INTER_CUBIC);
      cv::resize(im1, im1, cv::Size(kNorbResize, kNorbResize), 0, 0,
                 cv::INTER_CUBIC);
      // cv::namedWindow("after c1", CV_WINDOW_NORMAL );// Create a window for
      // display.
      // cv::namedWindow("after c2", CV_WINDOW_NORMAL );// Create a window for
      // display.
      // cv::imshow("after c1", im0);
      // cv::imshow("after c2", im1);
      // cv::waitKey(0);
      mat_to_datum(im0, im1, &datum_sm);
      // datum.set_data(pixels, kNorbImagesNBytes);
      datum_sm.set_label(label);
      datum_sm.SerializeToString(&value);
      snprintf(key, kMaxKeyLength, "%08d", batch_id * kNorbBatchSize + itemid);
      db->Put(leveldb::WriteOptions(), string(key), value);
    }
  } // end of reading train_db

  delete[] pixels;
}

void convert_dataset(const string &input_folder, const string &output_folder) {
  // leveldb options
  leveldb::Options options;
  options.create_if_missing = true;
  options.error_if_exists = true;

  // Open training leveldb
  leveldb::DB *train_db;
  leveldb::Status status =
      leveldb::DB::Open(options, output_folder + kTrainFilename, &train_db);
  CHECK(status.ok()) << "Failed to open leveldb "
                     << output_folder + kTrainFilename
                     << ". Is it already existing?";
  // Open test leveldb
  leveldb::DB *test_db;
  status = leveldb::DB::Open(options, output_folder + kTestFilename, &test_db);
  CHECK(status.ok()) << "Failed to open leveldb "
                     << output_folder + kTestFilename
                     << ". Is it already existing?";

  LOG(INFO) << "Writing training data";

  store_norb_batches(input_folder, kNorbTrainName, kNorbTrainBatches, train_db);

  LOG(INFO) << "Writing test data";
  store_norb_batches(input_folder, kNorbTestName, kNorbTestBatches, test_db);

  delete train_db;
  delete test_db;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("This script converts the NORB dataset to the leveldb format used\n"
           "by caffe to perform classification.\n"
           "Usage:\n"
           "    convert_norb_data input_folder output_folder\n"
           "The NORB-v1.0 dataset could be downloaded at\n"
           "  http://www.cs.nyu.edu/~ylclab/data/norb-v1.0/\n"
           "You should gunzip them after downloading.\n");
  } else {
    google::InitGoogleLogging(argv[0]);
    convert_dataset(string(argv[1]), string(argv[2]));
  }
  return 0;
}
