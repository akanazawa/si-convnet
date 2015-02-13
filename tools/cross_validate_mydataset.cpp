//
// Script for cross validating mydataset with various hyperparameters with
// specified basic model in proto
// Usage:
//    cross_validate_mnist base_solver_proto  model[cnn/sicnn/ricnn]

#include <cuda_runtime.h>
#include <fstream>
#include <time.h>

#include <cstring>
#include <cmath>

#include "boost/filesystem.hpp"

#include "caffe/caffe.hpp"
#include "caffe/util/imshow.hpp"

using namespace caffe; // NOLINT(build/namespaces)
using std::string;
// {$training_set}_{$model}_{$training_size}_{$filter_size1}C{$filtermap_1}_{filter_size2}C{$filtermap_2}_lr{$learning_rate}_wc{$weight_decay}_mom{$momentum}
const string kModelName = "%s_%s_%s_%dC%d_%dC%d_lr%g_wc%g_mom%g";

// beginning of transformation
const string kTransformationName = "_T%.1f-%.1f_on_C1-C2";
const string kSnapshotDir = "snapshot/";
const string kImageDir = "images/";
const string kProtoDir = "protos/";

const int kMaxBuffer = 600;
const int kDefaultSeed = 1701;

const string kDataDir = "../../data/mydataset/";
const string kTrainFile = kDataDir + "%s_list_%s.txt";
const string kTestFile = kDataDir + "%s_list_%s.txt";

// 0 is mnist, 1 is mnist-sc, 2 is mnist-rot
vector<float> best_scores(3, 100);
vector<string> best_names(3);

template <typename Dtype> vector<Dtype> initialize_vector(Dtype *array, int n) {
  vector<Dtype> vec(array, array + n);
  return vec;
}
template vector<string> initialize_vector(string *array, int n);
template vector<int> initialize_vector(int *array, int n);
template vector<float> initialize_vector(float *array, int n);

string set_configuration(const string &model, const string &training_size,
                         const string &training_set, const float &learning_rate,
                         const float &weight_decay, const float &momentum,
                         const int &feature_map1, const int &feature_map2,
                         const int &filter_size1, const int &filter_size2,
                         const vector<float> &transformation,
                         const int &random_seed, SolverParameter &solver_param,
                         NetParameter &train_net_param,
                         NetParameter &test_net_param) {
  char str_buffer[kMaxBuffer];

  V0LayerParameter *source_layer_param =
      train_net_param.mutable_layers(0)->mutable_layer();
  CHECK_EQ(source_layer_param->name(), "mydataset");
  V0LayerParameter *conv1_layer_param =
      train_net_param.mutable_layers(1)->mutable_layer();
  CHECK_EQ(conv1_layer_param->name(), "conv1");
  V0LayerParameter *conv2_layer_param =
      train_net_param.mutable_layers(5)->mutable_layer();
  CHECK_EQ(conv2_layer_param->name(), "conv2");

  V0LayerParameter *test_source_layer_param =
      test_net_param.mutable_layers(0)->mutable_layer();
  CHECK_EQ(test_source_layer_param->name(), "mydataset");
  V0LayerParameter *test_conv1_layer_param =
      test_net_param.mutable_layers(1)->mutable_layer();
  CHECK_EQ(test_conv1_layer_param->name(), "conv1");
  V0LayerParameter *test_conv2_layer_param =
      test_net_param.mutable_layers(5)->mutable_layer();
  CHECK_EQ(test_conv2_layer_param->name(), "conv2");

  // turn on TIconv
  if (model.compare("sicnn") == 0) {
    conv1_layer_param->set_type("TIconv");
    conv2_layer_param->set_type("TIconv");
    test_conv1_layer_param->set_type("TIconv");
    test_conv2_layer_param->set_type("TIconv");
  }

  // Net stuff:
  // data source:
  if (training_size.compare("60k") == 0) {
    snprintf(str_buffer, kMaxBuffer, kTrainFile.c_str(), "trainval",
             training_set.c_str());
    source_layer_param->set_source(str_buffer);
    snprintf(str_buffer, kMaxBuffer, kTestFile.c_str(), "test",
             training_set.c_str());
    test_source_layer_param->set_source(str_buffer);
    solver_param.set_snapshot(5);
  } else if (training_size.compare("10k") == 0) {
    snprintf(str_buffer, kMaxBuffer, kTrainFile.c_str(), "batch_1",
             training_set.c_str());
    source_layer_param->set_source(str_buffer);
    snprintf(str_buffer, kMaxBuffer, kTestFile.c_str(), "test",
             training_set.c_str());
    test_source_layer_param->set_source(str_buffer);
    solver_param.set_snapshot(25);
  } else {
    snprintf(str_buffer, kMaxBuffer, kTrainFile.c_str(), "train",
             training_set.c_str());
    source_layer_param->set_source(str_buffer);
    snprintf(str_buffer, kMaxBuffer, kTestFile.c_str(), "valid",
             training_set.c_str());
    test_source_layer_param->set_source(str_buffer);
    solver_param.set_snapshot(10);
  }

  // feature map:
  conv1_layer_param->set_num_output(feature_map1);
  test_conv1_layer_param->set_num_output(feature_map1);

  conv2_layer_param->set_num_output(feature_map2);
  test_conv2_layer_param->set_num_output(feature_map2);

  // filter size:
  conv1_layer_param->set_kernelsize(filter_size1);
  test_conv1_layer_param->set_kernelsize(filter_size1);

  conv2_layer_param->set_kernelsize(filter_size2);
  test_conv2_layer_param->set_kernelsize(filter_size2);

  float tstart, tend;
  if (model.compare("sicnn") == 0) {
    conv1_layer_param->clear_transformations();
    conv2_layer_param->clear_transformations();
    test_conv1_layer_param->clear_transformations();
    test_conv2_layer_param->clear_transformations();

    tstart = transformation[0];
    tend = transformation[transformation.size() - 1];

    for (int t = 0; t < transformation.size() + 1; ++t) {
      if (t == 0) {
        // 0th is always the identity that's it (unless border/NN|bilinear is to
        // be set).
      } else {
        TransParameter *tmp = conv1_layer_param->add_transformations();
        TransParameter *test_tmp =
            test_conv1_layer_param->add_transformations();
        tmp->set_scale(transformation[t - 1]);
        test_tmp->set_scale(transformation[t - 1]);
        if (transformation[t - 1] >= 0.5) {
          TransParameter *tmp2 = conv2_layer_param->add_transformations();
          TransParameter *test_tmp2 =
              test_conv2_layer_param->add_transformations();
          tmp2->set_scale(transformation[t - 1]);
          test_tmp2->set_scale(transformation[t - 1]);
        }
      }
    } // end for transformation
  }

  // Solver stuff:
  int epoch_size;
  if (training_size.compare("3k") == 0) {
    epoch_size = floor(3000 / source_layer_param->batchsize());
  } else if (training_size.compare("5k") == 0) {
    epoch_size = floor(5000 / source_layer_param->batchsize());
  } else if (training_size.compare("10k") == 0) {
    epoch_size = floor(10000 / source_layer_param->batchsize());
  } else if (training_size.compare("30k") == 0) {
    epoch_size = floor(30000 / source_layer_param->batchsize());
  } else if (training_size.compare("50k") == 0) {
    epoch_size = floor(50000 / source_layer_param->batchsize());
  } else if (training_size.compare("60k") == 0) {
    epoch_size = floor(60000 / source_layer_param->batchsize());
    LOG(INFO) << "Using all training data!";
  } else {
    LOG(FATAL) << "Unknown training size!";
  }
  // Set epoch_size based on the training size!
  solver_param.set_epoch_size(epoch_size);

  solver_param.set_base_lr(learning_rate);

  solver_param.set_weight_decay(weight_decay);

  solver_param.set_momentum(momentum);

  // cmodel name:
  snprintf(str_buffer, kMaxBuffer, kModelName.c_str(), training_set.c_str(),
           model.c_str(), training_size.c_str(), filter_size1, feature_map1,
           filter_size2, feature_map2, learning_rate, weight_decay, momentum);
  string model_name(str_buffer);

  if (random_seed != kDefaultSeed) {
    snprintf(str_buffer, kMaxBuffer, "%d", random_seed);
    model_name += "_seed" + string(str_buffer);
  }

  if (model.compare("sicnn") == 0) {
    snprintf(str_buffer, kMaxBuffer, kTransformationName.c_str(), tstart, tend);
    model_name = model_name + str_buffer;
  }
  // set snapshot name
  solver_param.set_snapshot_prefix(kSnapshotDir + model + "/" + model_name);

  // Check:
  CHECK_EQ(learning_rate, solver_param.base_lr());
  CHECK_EQ(weight_decay, solver_param.weight_decay());

  CHECK_EQ(feature_map1, conv1_layer_param->num_output());
  CHECK_EQ(feature_map1, test_conv1_layer_param->num_output());

  CHECK_EQ(feature_map2, conv2_layer_param->num_output());
  CHECK_EQ(feature_map2, test_conv2_layer_param->num_output());

  CHECK_EQ(filter_size1, conv1_layer_param->kernelsize());
  CHECK_EQ(filter_size1, test_conv1_layer_param->kernelsize());

  CHECK_EQ(filter_size2, conv2_layer_param->kernelsize());
  CHECK_EQ(filter_size2, test_conv2_layer_param->kernelsize());

  if (model.compare("sicnn") == 0) {
    CHECK_EQ(1, conv1_layer_param->transformations(0).scale());
    CHECK_EQ(1, test_conv1_layer_param->transformations(0).scale());

    CHECK_EQ(tstart, conv1_layer_param->transformations(1).scale());
    CHECK_EQ(tend,
             conv1_layer_param->transformations(transformation.size()).scale());

    CHECK_EQ(tstart, test_conv1_layer_param->transformations(1).scale());
    CHECK_EQ(tend, test_conv1_layer_param->transformations(
                                               transformation.size()).scale());
  }

  // summary:
  LOG(INFO) << "********************";
  LOG(INFO) << "***** Model:" << model << " training size: " << training_size
            << " (epoch-size: " << solver_param.epoch_size() << ")";
  LOG(INFO) << "***** Training source: " << source_layer_param->source()
            << " test source: " << test_source_layer_param->source();
  LOG(INFO) << "***** Learning rate: " << learning_rate
            << " weight decay: " << weight_decay
            << " feature maps: " << feature_map1 << ", " << feature_map2
            << " filter sizes: " << filter_size1 << ", " << filter_size2;

  if (model.compare("sicnn") == 0)
    LOG(INFO) << "***** Using transformation " << tstart << " to " << tend;

  LOG(INFO) << "***** Saving with name " << model_name;
  LOG(INFO) << "********************";

  return model_name;
}

bool file_exists(const char *filename) {
  std::ifstream infile(filename);
  return infile.good();
}

string int2string(int val) {
  char str_buffer[kMaxBuffer];
  snprintf(str_buffer, kMaxBuffer, "%d", val);
  return string(str_buffer);
}

float test_net(const NetParameter &trained_net,
               const NetParameter &test_net_param, int total_iter) {
  Caffe::set_phase(Caffe::TEST);
  Net<float> *test_net = new Net<float>(test_net_param, true);
  test_net->CopyTrainedLayersFrom(trained_net);

  LOG(INFO) << "Running " << total_iter << " Iterations";
  int show_every = total_iter / 2;
  double test_accuracy = 0;
  vector<Blob<float> *> dummy_blob_input_vec;
  for (int i = 0; i < total_iter; ++i) {
    const vector<Blob<float> *> &result =
        test_net->Forward(dummy_blob_input_vec);
    test_accuracy += result[0]->cpu_data()[0];
    if (i % show_every == 0)
      LOG(INFO) << "Batch " << i << ", accuracy: " << result[0]->cpu_data()[0];
  }
  test_accuracy /= total_iter;

  LOG(INFO) << "Test accuracy:" << test_accuracy
            << " test error (%):" << 100 - test_accuracy * 100;
  printf("test accuracy/error: %f %f\n", test_accuracy,
         100 - test_accuracy * 100);
  // return error
  delete test_net;
  return (100 - test_accuracy * 100);
}

void train_with_setting(const SolverParameter &solver_param,
                        const NetParameter &train_net_param,
                        NetParameter &test_net_param, const string &model_name,
                        int seed) {
  Caffe::set_phase(Caffe::TRAIN);
  // reset the seed
  Caffe::set_random_seed(seed);
  std::srand(seed);
  LOG(INFO) << "using rand seed " << seed;
  string snapshot_name = solver_param.snapshot_prefix() + "_epoch_" +
                         int2string(solver_param.max_iter());
  string image_name =
      kImageDir + model_name + "_epoch_" + int2string(solver_param.max_iter());

  NetParameter trained_net_param;

  if (file_exists(snapshot_name.c_str())) {
    LOG(INFO) << "--- skipping training, " << snapshot_name
              << " already exists";
    ReadProtoFromBinaryFile(snapshot_name, &trained_net_param);
  } else {
    // auto generate the current train/test_net and save it
    string proto_name = kProtoDir + "train_" + model_name;
    if (!file_exists(proto_name.c_str())) {
      LOG(INFO) << "Writing current config to protofile as " << proto_name;
      WriteProtoToTextFile(train_net_param, proto_name);
      proto_name = kProtoDir + "test_" + model_name;
      WriteProtoToTextFile(test_net_param, proto_name);
      proto_name = kProtoDir + "solver_" + model_name;
      WriteProtoToTextFile(solver_param, proto_name);
    }
    // TODO: put things in try catch in case some settings throw exception
    time_t start_t, end_t;
    time(&start_t);

    LOG(INFO) << "---------- Starting Optimization on " << model_name
              << " ----------";
    // If sicnn, don't do test because there arent enough memory
    // SGDSolver<float> solver(solver_param, train_net_param, test_net_param);
    SGDSolver<float> solver(solver_param);
    solver.Solve();

    time(&end_t);
    double elapsed_secs = difftime(end_t, start_t);
    LOG(INFO) << "---------- Optimization on " << model_name << " done "
              << elapsed_secs / (3600) << " hrs. ----------";

    shared_ptr<Net<float> > trained_net = solver.net();
    trained_net->ToProto(&trained_net_param);

    vector<shared_ptr<Blob<float> > > &layer_blobs =
        trained_net->layers()[1]->blobs();
    CHECK_EQ(trained_net->layer_names()[1], "conv1");

    LOG(INFO) << "--- Saving filter image to " << image_name;
    save_montage(layer_blobs[0].get(), image_name + "_W1");
  }

  // Test on other testsets:
  V0LayerParameter *test_source_layer_param =
      test_net_param.mutable_layers(0)->mutable_layer();
  CHECK_EQ(test_source_layer_param->name(), "mydataset");

  string testsets[] = { "valid_list_lcn-mean_centered_51_25.txt",
                        "test_list_lcn-mean_centered_51_25.txt",
                        "test_list_lcn-mean_centered.txt" };

  int num_test = sizeof(testsets) / sizeof(string);
  for (int i = 0; i < num_test; ++i) {
    test_source_layer_param->set_source(kDataDir + testsets[i]);
    LOG(INFO) << "--- Testing on " << test_source_layer_param->source();
    test_net(trained_net_param, test_net_param, 100);
  }
}

int main(int argc, char **argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc != 3) {
    LOG(ERROR)
        << "Usage: cross_validate_mydataset base_solver_proto model[cnn/sicnn]";
    return 0;
  }

  // this is platform dependent..
  CHECK(file_exists(kSnapshotDir.c_str())) << kSnapshotDir << " doesnt exist!";
  CHECK(file_exists(kImageDir.c_str())) << kImageDir << " doesnt exist!";
  CHECK(file_exists(kProtoDir.c_str())) << kProtoDir << " doesnt exist!";

  SolverParameter base_solver_param;
  ReadProtoFromTextFile(argv[1], &base_solver_param);

  LOG(INFO) << "Creating base training net from "
            << base_solver_param.train_net();
  NetParameter train_net_param;
  ReadProtoFromTextFile(base_solver_param.train_net(), &train_net_param);

  LOG(INFO) << "Creating base testing net from "
            << base_solver_param.test_net(0);
  NetParameter test_net_param;
  ReadProtoFromTextFile(base_solver_param.test_net(0), &test_net_param);

  string model(argv[2]);

  // Hyper parameters
  // string tr_size[] = {"50k"};
  string tr_size[] = { "60k" };
  // string tr_size[] = {"10k"};
  const vector<string> training_size =
      initialize_vector(tr_size, sizeof(tr_size) / sizeof(string));
  // "whitened_51_25",
  // string tr_set[] = { "lcn-mean_centered_51_25", "lcn-mean_centered" };
  string tr_set[] = { "lcn-mean_centered_51_25" };
  const vector<string> training_set =
      initialize_vector(tr_set, sizeof(tr_set) / sizeof(string));

  // int fmap1[] = { 9 };
  // int fmap2[] = { 9 };
  int fmap1[] = { 96 };
  int fmap2[] = { 96 };

  vector<vector<int> > feature_map(2);
  feature_map[0] = initialize_vector(fmap1, sizeof(fmap1) / sizeof(int));
  feature_map[1] = initialize_vector(fmap2, sizeof(fmap2) / sizeof(int));

  // int fsize1[] = { 9 };
  // int fsize2[] = { 5 };
  int fsize1[] = { 7 };
  int fsize2[] = { 5 };

  vector<vector<int> > filter_size(2);
  filter_size[0] = initialize_vector(fsize1, sizeof(fsize1) / sizeof(int));
  filter_size[1] = initialize_vector(fsize2, sizeof(fsize2) / sizeof(int));

  // float blr[] = { 0.0005, 0.001 };
  float blr[] = { 0.001, 0.005 };
  const vector<float> base_learning_rate =
      initialize_vector(blr, sizeof(blr) / sizeof(float));

  // float wc[] = { 0.005, 0.001 };
  float wc[] = { 0.005, 0.001 };
  const vector<float> weight_decay =
      initialize_vector(wc, sizeof(wc) / sizeof(float));

  // float mom[] = { 0.9, 0.7 };
  float mom[] = { 0.9, 0.7, 0.5 };
  const vector<float> momentum =
      initialize_vector(mom, sizeof(mom) / sizeof(float));

  // transformations
  vector<vector<float> > transformation;
  if (model.compare("sicnn") == 0) {
    // 2^(1/3) = 1.26 step size
    // float t2 [] = { 0.63, 0.7937, 1.2599, 1.5874, 2.};
    // transformation.push_back( initialize_vector(t2, sizeof(t2)/sizeof(float))
    // );

    // // 1.2 stepsize
    // float t3 [] = { 0.63, 0.7937, 1.2599, 1.5874};
    // transformation.push_back( initialize_vector(t3, sizeof(t3)/sizeof(float))
    // );
    // 1.2 stepsize
    float t4[] = { 0.6944, 0.8333, 1.2, 1.44, 1.728 };
    transformation.push_back(initialize_vector(t4, sizeof(t4) / sizeof(float)));
  }

  // int rs[] = { 1701, 1216, 1001, 818, 426 };
  int rs[] = { 1701 };
  const vector<int> random_seeds =
      initialize_vector(rs, sizeof(rs) / sizeof(int));

  Caffe::set_mode(Caffe::Brew(base_solver_param.solver_mode()));

  // Start for loop..

  string my_tsize, my_tset;
  int my_fmap1, my_fmap2, my_fsize1, my_fsize2, my_randseed;
  vector<float> my_transformation;
  float my_lr, my_wc, my_mom;
  for (int r = 0; r < random_seeds.size(); ++r) {
    my_randseed = random_seeds[r];
    for (int tsize = 0; tsize < training_size.size(); ++tsize) {
      my_tsize = training_size[tsize];

      for (int tset = 0; tset < training_set.size(); ++tset) {
        my_tset = training_set[tset];

        for (int fmap = 0; fmap < feature_map[0].size(); ++fmap) {
          my_fmap1 = feature_map[0][fmap];
          my_fmap2 = feature_map[1][fmap];

          for (int fsize = 0; fsize < filter_size[0].size(); ++fsize) {
            my_fsize1 = filter_size[0][fsize];
            my_fsize2 = filter_size[1][fsize];

            for (int lr = 0; lr < base_learning_rate.size(); ++lr) {
              my_lr = base_learning_rate[lr];

              for (int wd = 0; wd < weight_decay.size(); ++wd) {
                my_wc = weight_decay[wd];

                for (int mome = 0; mome < momentum.size(); ++mome) {
                  my_mom = momentum[mome];

                  if (model.compare("sicnn") == 0) {
                    for (int t = 0; t < transformation.size(); ++t) {
                      my_transformation = transformation[t];

                      string model_name = set_configuration(
                          model, my_tsize, my_tset, my_lr, my_wc, my_mom,
                          my_fmap1, my_fmap2, my_fsize1, my_fsize2,
                          my_transformation, my_randseed, base_solver_param,
                          train_net_param, test_net_param);

                      LOG(INFO)
                          << " bc SI-ConvNet, not testing during training time";
                      base_solver_param.clear_test_net();
                      base_solver_param.clear_test_iter();
                      base_solver_param.clear_test_interval();

                      train_with_setting(base_solver_param, train_net_param,
                                         test_net_param, model_name,
                                         my_randseed);
                    } // end of transformations
                  } else {
                    string model_name = set_configuration(
                        model, my_tsize, my_tset, my_lr, my_wc, my_mom,
                        my_fmap1, my_fmap2, my_fsize1, my_fsize2,
                        my_transformation, my_randseed, base_solver_param,
                        train_net_param, test_net_param);

                    train_with_setting(base_solver_param, train_net_param,
                                       test_net_param, model_name, my_randseed);
                  } // end checking model
                }   // end momentum
              }     // end weight decay
            }       // end lr
          }
        }
      }
    }
  }

  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
