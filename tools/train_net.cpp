// Copyright 2013 Yangqing Jia
//
// This is a simple script that allows one to quickly train a network whose
// parameters are specified by text format protocol buffers.
// Usage:
//    train_net net_proto_file solver_proto_file [resume_point_file]

#include <cuda_runtime.h>
#include <time.h>

#include <cstring>

#include "caffe/caffe.hpp"

int main(int argc, char** argv) {
  // Caffe::set_random_seed(1701);
  // std::srand(1701);

  // ::google::InitGoogleLogging(argv[0]);
  // if (argc < 2) {
  //   LOG(ERROR) << "Usage: train_net solver_proto_file [resume_point_file]";
  //   return 0;
  // }

  // SolverParameter solver_param;
  // ReadProtoFromTextFile(argv[1], &solver_param);

  // time_t start_t, end_t;
  // time(&start_t);

  // LOG(INFO) << "---------- Starting Optimization with "
  // 	    << "net specified in " << argv[1] << " ----------";
  // SGDSolver<float> solver(solver_param);
  // if (argc == 3) {
  //   LOG(INFO) << "Resuming from " << argv[2];
  //   solver.Solve(argv[2]);
  // } else {
  //   solver.Solve();
  // }

  // time(&end_t);
  // double elapsed_secs = difftime(end_t, start_t);
  // LOG(INFO) << "---------- Optimization Done " <<
  //   elapsed_secs/(3600) << " hrs. ----------";

  // google::protobuf::ShutdownProtobufLibrary();

  LOG(FATAL) << "Deprecated. Use caffe train --solver=... "
                "[--snapshot=...] instead.";
  return 0;
}
