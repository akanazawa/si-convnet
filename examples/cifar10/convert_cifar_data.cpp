//
// This script converts the CIFAR dataset to the leveldb format used
// by caffe to perform classification.
// Usage:
//    convert_cifar_data input_folder output_db_file
// The CIFAR dataset could be downloaded at
//    http://www.cs.toronto.edu/~kriz/cifar.html
//
//AJ: Edited it so that 1-5 batch goes to trainval-leveldb
//    1-4 goes to train-leveldb, 5 goes to val-leveldb

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "leveldb/db.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"

using std::string;


const int kCIFARSize = 32;
const int kCIFARImageNBytes = 3072;
const int kCIFARBatchSize = 10000;
const int kCIFARTrainBatches = 5;

void read_image(std::ifstream* file, int* label, char* buffer) {

  char label_char;
  file->read(&label_char, 1);
  *label = label_char;
  file->read(buffer, kCIFARImageNBytes);
  return;
}

void convert_dataset(const string& input_folder, const string& output_folder) {
  // Leveldb options
  leveldb::Options options;
  options.create_if_missing = true;
  options.error_if_exists = true;
  // Data buffer
  int label;
  char str_buffer[kCIFARImageNBytes];
  string value;
  caffe::Datum datum;
  datum.set_channels(3);
  datum.set_height(kCIFARSize);
  datum.set_width(kCIFARSize);

  string trainval_filename = "/cifar-trainval-leveldb";
  string train_filename = "/cifar-train-leveldb";   
  string val_filename = "/cifar-val-leveldb";   

  LOG(INFO) << "Writing Training+Val data";
  leveldb::DB* trainval_db;
  leveldb::Status status_trainval = leveldb::DB::Open(options, 
						      output_folder + trainval_filename,
						      &trainval_db);
  CHECK(status_trainval.ok()) << "Failed to open trainval leveldb.";

  leveldb::DB* train_db;
  leveldb::Status status_train = leveldb::DB::Open(options, 
						   output_folder + train_filename,
						   &train_db);
  CHECK(status_train.ok()) << "Failed to open train leveldb.";

  leveldb::DB* val_db;
  leveldb::Status status_val = leveldb::DB::Open(options, 
						   output_folder + val_filename,
						   &val_db);
  CHECK(status_val.ok()) << "Failed to open val leveldb.";

  for (int fileid = 0; fileid < kCIFARTrainBatches; ++fileid) {
    // Open files
    LOG(INFO) << "Training Batch " << fileid + 1;
    snprintf(str_buffer, kCIFARImageNBytes, "/data_batch_%d.bin", fileid + 1);
    std::ifstream data_file((input_folder + str_buffer).c_str(),
        std::ios::in | std::ios::binary);
    CHECK(data_file) << "Unable to open train file #" << fileid + 1;

    for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
      read_image(&data_file, &label, str_buffer);
      // ----- debug show with opencv
      // cv::Mat image(kCIFAR_SIZE, kCIFAR_SIZE, CV_8UC1, str_buffer, 0);
      // LOG(INFO) << image.rows << " by " << image.cols << " by " << image.channels() ;
      // cv::namedWindow( "Display window", CV_WINDOW_NORMAL );// Create a window for display.
      // cv::imshow( "Display window", image );
      // cv::waitKey(0);
      // ----- end debug
      datum.set_label(label);
      datum.set_data(str_buffer, kCIFARImageNBytes);
      datum.SerializeToString(&value);

      sprintf(str_buffer, "%05d", fileid * kCIFARBatchSize + itemid);
      trainval_db->Put(leveldb::WriteOptions(), string(str_buffer), value);
      if (fileid == (kCIFARTrainBatches-1) ) {
        val_db->Put(leveldb::WriteOptions(), string(str_buffer), value);
      } else {
        train_db->Put(leveldb::WriteOptions(), string(str_buffer), value);
      }

    }
  }

  LOG(INFO) << "Writing Testing data";
  leveldb::DB* test_db;
  CHECK(leveldb::DB::Open(options, output_folder + "/cifar10_test_leveldb",
      &test_db).ok()) << "Failed to open leveldb.";
  // Open files
  std::ifstream data_file((input_folder + "/test_batch.bin").c_str(),
      std::ios::in | std::ios::binary);
  CHECK(data_file) << "Unable to open test file.";
  for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
    read_image(&data_file, &label, str_buffer);
    datum.set_label(label);
    datum.set_data(str_buffer, kCIFARImageNBytes);
    datum.SerializeToString(&value);
    snprintf(str_buffer, kCIFARImageNBytes, "%05d", itemid);
    test_db->Put(leveldb::WriteOptions(), string(str_buffer), value);
  }

  delete trainval_db;
  delete train_db;
  delete val_db;
  delete test_db;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("This script converts the CIFAR dataset to the leveldb format used\n"
           "by caffe to perform classification.\n"
           "Usage:\n"
           "    convert_cifar_data input_folder output_folder\n"
           "Where the input folder should contain the binary batch files.\n"
           "The CIFAR dataset could be downloaded at\n"
           "    http://www.cs.toronto.edu/~kriz/cifar.html\n"
           "You should gunzip them after downloading.\n");
  } else {
    google::InitGoogleLogging(argv[0]);
    convert_dataset(string(argv[1]), string(argv[2]));
  }
  return 0;
}
