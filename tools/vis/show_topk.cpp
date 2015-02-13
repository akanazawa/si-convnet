// Angjoo Kanazawa
// show content of leveldb as caffe sees it
// Usage:
//    visualize_leveldb input_leveldb_file

#include <iostream>
#include <fstream>
#include <string>
#include <queue>
#include <functional>

#include <glog/logging.h>
#include <leveldb/db.h>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/util/imshow.hpp"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/caffe.hpp"

using std::string;
using std::priority_queue;
using namespace caffe;  // NOLINT(build/namespaces)

void load_net(const char* net_proto, const char* trainednet_proto, Net<float>* net) {

  NetParameter test_net_param;
  ReadProtoFromTextFile(net_proto, &test_net_param);
  net = new Net<float>(test_net_param);

  NetParameter trained_net_param;
  ReadProtoFromBinaryFile(trainednet_proto, &trained_net_param);
  net->CopyTrainedLayersFrom(trained_net_param);

}
void open_leveldb(const char* db_filename, leveldb::Iterator* it, int& n_test) {
  // open leveldb
  leveldb:: DB* db_orig;
  leveldb::Options options;
  options.create_if_missing = false;
  options.error_if_exists = false;
  leveldb::Status status_orig = leveldb::DB::Open(options, db_filename, &db_orig);
  CHECK(status_orig.ok()) << "Failed to open leveldb " << db_filename;
  // get total number of elements in this db (can't seem to find that function)
  std::vector<string> keys;
  it = db_orig->NewIterator(leveldb::ReadOptions());
  CHECK(it->status().ok());  // Check for any errors found during the scan

  n_test = 0;
  // leveldb doesn't have size() function..?
  for (it->SeekToFirst(); it->Valid(); it->Next(), ++n_test);
  LOG(INFO) << " testing on " << n_test << " images";
}

void set_input_size(leveldb::Iterator* it, Net<float>*net, int n_batch) {
  caffe::Datum datum;
  it->SeekToFirst();
  datum.ParseFromString(it->value().ToString());
  vector<Blob<float>* >& input_blobs = net->input_blobs();
  CHECK(input_blobs.size() == 1);
  input_blobs[0]->Reshape(n_batch, datum.channels(), datum.height(), datum.width());
}

void set_input(leveldb::Iterator* it, float* img_data) { 

  caffe::Datum datum;
  datum.ParseFromString(it->value().ToString());
  const string& data = datum.data();
  int size = datum.channels() * datum.height() * datum.width();
  if (data.size()) {
    for (int j = 0; j < size; ++j ) {
      img_data[j] = static_cast<float>((uint8_t)data[j]);
    }
  } else {
    for (int j = 0; j < size; ++j) {
      img_data[j] = datum.float_data(j);
    }
  }
  CHECK(it->status().ok());  // Check for any errors found during the scan
}

void show_topk(const char* net_proto, const char* trained_net_proto, 
	       const char* db_filename, string layer_name, int k) {
  std::ifstream db_file(db_filename, std::ios::binary);
  CHECK(db_file) << "leveldb file doesn't exist " << db_filename;
  
  // initialize everything
  Net<float>* net;
  int n_test = 0;
  int layer_id = -1;
  int neuron_id = 0;
  leveldb::Iterator* it;

  // min-heap
  priority_queue<float, std::vector<float>, std::greater<float> > heap;
  
  // load net
  load_net(net_proto, trained_net_proto, net);

  // load leveldb
  open_leveldb(db_filename, it, n_test);

  // find the layer_id
  vector<string> layer_names = net->layer_names();
  for (int i = 0; i < layer_names.size(); ++i) {
    if ( layer_name.compare(layer_names[i]) == 0 ) {
      layer_id = i;
      break;
    }
  }   
  if ( layer_id == -1 ) {
    LOG(ERROR) << "Can't find layer name " << layer_name;
    exit(1);
  } else { // check that activation exists
    CHECK( net->top_vecs().size() >= layer_id );
  }
  // set target activation:    
  // AJ: assuming convolution layer 
  Blob<float>* target_blob = net->top_vecs()[layer_id][0];
  int height = target_blob->height();
  int sheet_size = height*target_blob->width();
  CHECK(neuron_id < target_blob->channels() ) << "neuron_id: " << neuron_id <<
    ", there are " << target_blob->channels() << " neurons";
  
  // reset the input size
  set_input_size(it, net, 1);
  Blob<float>* input_blob = net->input_blobs()[0];

  int h, w, r0, c0, r1, c1;
  // start testing: doing 1 by 1 for now..
  for (it->SeekToFirst(); it->Valid(); it->Next()) {    
    // 0. set input
    set_input(it, input_blob->mutable_cpu_data());
    // 1. forward pass
    net->ForwardPrefilled();
    // 2. get the activation:
    const float* score = target_blob->cpu_data() + target_blob->offset(1, neuron_id, 0, 0);
    for (int j = 0; j < sheet_size; ++j) {
      // add in heap
      if ( heap.size() < k || score[j] >= heap.top() ) { 
      	if ( heap.size() == k ) heap.pop();	
	heap.push(score[j]);
	// get image at this place
	h = j / height;
	w = j % height; 
	r0 = h;	r1 = h;
	c0 = w; c1 = w;
	// prop_region(net, layer_id, r0, c0, r1, c1);
	//r0, c0 is the left top point, r1, c1 is the right bottom point
	// crop_image(input_blob, r0, c0, r1, c1);
      }
    }
  }// end of for(it->SeekToFirst()


  // clean up
  delete it;
  delete net;
}

int main (int argc, char** argv) {
  if (argc < 5 ) {
    printf("Usage:\n"
           "   show_topk.bin net_proto trained_net_proto input_leveldb_file layer_name [k] [CPU/GPU]\n");    
  } else {
    int k = argc < 5 ? 10 : atoi(argv[5]);
    if ( argc == 7 && strcmp(argv[6], "GPU") == 0 ) {
      LOG(INFO) << "Using GPU";
      Caffe::set_mode(Caffe::GPU);
    } else {
      LOG(INFO) << "Using CPU";
      Caffe::set_mode(Caffe::CPU);
    }

    string layer_name(argv[3]);
    google::InitGoogleLogging(argv[0]);
    show_topk(argv[1], argv[2], argv[3], layer_name, k);
  }
  return 0;
}
