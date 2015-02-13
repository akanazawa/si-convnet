// Copyright 2013 Yangqing Jia
//
// This is a simple script that allows one to quickly finetune a network.
// Usage:
//    finetune_net solver_proto_file pretrained_net

#include <cuda_runtime.h>
#include <time.h>

#include <string>

#include "caffe/caffe.hpp"

int main(int argc, char** argv) {

  // ::google::InitGoogleLogging(argv[0]);
  // if (argc < 2) {
  //   LOG(ERROR) << "Usage: finetune_net solver_proto_file pretrained_net";
  //   return 0;
  // }

  // SolverParameter solver_param;
  // ReadProtoFromTextFile(argv[1], &solver_param);

  // time_t start_t, end_t;
  // time(&start_t);

  // LOG(INFO) << "---------- Starting finetuning with "
  // 	    << "net specified in " << argv[1] << " ----------";

  // SGDSolver<float> solver(solver_param);
  // LOG(INFO) << "Loading from " << argv[2];
  // solver.net()->CopyTrainedLayersFrom(string(argv[2]));
  // solver.Solve();

  // time(&end_t);
  // double elapsed_secs = difftime(end_t, start_t);
  // LOG(INFO) << "---------- Optimization Done " <<
  //   elapsed_secs/(3600) << " hrs. ----------";

  LOG(FATAL) << "Deprecated. Use caffe train --solver=... "
                "[--weights=...] instead.";
  return 0;
}
