// Angjoo Kanazawa
//
// This is a simple script to visualize the features learned 
// Usage:
//    visualize_net pretrained_net_proto [prefix]

#include <cstdio>
#include <vector>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/caffe.hpp"
#include "caffe/util/imshow.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::vector;


int main(int argc, char** argv) {
  if (argc < 2) {
    LOG(ERROR) << "Usage: " <<
      " visualize_net pretrained_net_proto [fname]" <<
      " if [fname] is set, saves the image.";
    return 0;
  }
  
  bool save = argc == 3 ? true : false;
  std::string fname = save ? std::string(argv[2]) : "";
  
  Caffe::set_phase(Caffe::TEST);
  Caffe::set_mode(Caffe::CPU);

  NetParameter trained_net_param;
  ReadProtoFromBinaryFile(argv[1], &trained_net_param);

  // copy learned contents into net
  Net<float>* net = new Net<float>(trained_net_param);
  net->CopyTrainedLayersFrom(trained_net_param);
  const vector<shared_ptr<Layer<float> > >& layers = net->layers();
  for (int i = 0; i < layers.size() ; ++i) {
    std::string layer_name = net->layer_names()[i];
    vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
    if ( layer_blobs.size() > 0 && layer_name.find("conv") != std::string::npos) {      
      // just do W
      if ( save ) {
	caffe::save_montage(layer_blobs[0].get(), fname + "_" + layer_name);
      } else {
	caffe::montage(layer_blobs[0].get(), layer_name);
      }
      // for (int j = 0; j < layer_blobs.size(); ++j) {
      //   snprintf(str_buffer, 300, "_param_%d", j);  
      //   caffe::montage(layer_blobs[j].get(), prefix+layer_name+str_buffer);
      // }
      cv::waitKey(0);
    }
  }

  // clean up
  delete net;

  return 0;
}
