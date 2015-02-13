// Angjoo Kanazawa
//
// This script splits a level-db with DATUM in it into 2 level-db, one training,
// one validation
// Usage:
//    split_training input_leveldb_file output_train_file output_val_file n_val

#include <fstream>
#include <iostream>
#include <ostream>
#include <cstdlib>
#include <algorithm>

#include <glog/logging.h>
#include <leveldb/db.h>

void split_dataset(const char *db_filename, const char *train_filename,
                   const char *valid_filename, const int &num_val) {
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
  keys.reserve(60000);
  leveldb::Iterator *it = db_orig->NewIterator(leveldb::ReadOptions());
  CHECK(it->status().ok()); // Check for any errors found during the scan
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    keys.push_back(it->key().ToString());
  }
  CHECK(it->status().ok()); // Check for any errors found during the scan

  int num_items = keys.size();
  LOG(INFO) << "Total of " << num_items << " items found.";

  CHECK(num_val < num_items) << "# of validation " << num_val
                             << " is too large";
  // randomly shuffle the key array.
  std::srand(1216);
  std::random_shuffle(keys.begin(), keys.end());

  // Take the first num_val and store it to validation db.
  // Make new db
  leveldb::DB *db_val;
  options.create_if_missing = true;
  options.error_if_exists = true;
  leveldb::Status status_val =
      leveldb::DB::Open(options, valid_filename, &db_val);
  CHECK(status_val.ok()) << "Failed to open leveldb " << valid_filename
                         << ". Is it already existing?";
  // make new train
  leveldb::DB *db_train;
  leveldb::Status status_tr =
      leveldb::DB::Open(options, train_filename, &db_train);
  CHECK(status_tr.ok()) << "Failed to open leveldb " << train_filename
                        << ". Is it already existing?";

  // SPLIT: put the first num_val into db_val, rest into db_train
  std::string value;
  for (int i = 0; i < num_items; ++i) {
    // std::cout << "Saving item with key " << keys[i] << " to ";
    db_orig->Get(leveldb::ReadOptions(), keys[i], &value);
    CHECK(status_orig.ok()) << "Failed to get item with key " << keys[i]
                            << " at " << i;
    if (i < num_val) {
      // std::cout << "validation set" << std::endl;
      db_val->Put(leveldb::WriteOptions(), keys[i], value);
      CHECK(status_val.ok()) << "Failed to save item to val with key "
                             << keys[i];
    } else {
      // std::cout << "train set" << std::endl;
      db_train->Put(leveldb::WriteOptions(), keys[i], value);
      CHECK(status_tr.ok()) << "Failed to save item with key " << keys[i];
    }
  }
  // check:
  int num_val_test = 0;
  it = db_val->NewIterator(leveldb::ReadOptions());
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    ++num_val_test;
    db_train->Get(leveldb::ReadOptions(), it->key().ToString(), &value);
    CHECK(~status_tr.ok()) << "Training and validation Contains duplicate key "
                           << it->key().ToString();
  }

  delete db_orig;
  delete db_val;
  delete db_train;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    printf("This script splits an existing level-db dataset with proto datum "
           "in it to 2 level-db files\n"
           "Usage:\n"
           "    split_training input_leveldb_file output_train_file "
           "output_val_file n_val\n");
  } else {
    google::InitGoogleLogging(argv[0]);
    split_dataset(argv[1], argv[2], argv[3], atoi(argv[4]));
  }
  return 0;
}
