// Benchmarking the speed up comes from calling a big dgemm once vs calling small dgemm T many times.

#include <cuda_runtime.h>

#include <cmath>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"

using namespace caffe;
using std::vector;

int main(int argc, char** argv) {

  // if (argc < 2) {
  //   LOG(ERROR) << "dgemm_benchmark [iterations=50] [T=6] [M=256] [input_size=51]";
  //   return 0;
  // }

  Caffe::set_mode(Caffe::GPU);

  int total_iter = 50;
  if (argc >= 2) {
    total_iter = atoi(argv[1]);
  }
  
  int T = 6;
  if (argc >=3) {
    T = atoi(argv[2]);
  }

  int M = 256; // size of feature maps
  if (argc >=4) {
    M = atoi(argv[3]);    
  }
  
  int ksize = 7; 
  int channels = 64;

  int input_size = 51; // input image size
  if (argc >= 5) {
    input_size = atoi(argv[4]);
  }

  int K = channels * ksize * ksize ;

  vector<int> Ns (T);  
  int total_N = 0;
  
  LOG(INFO) << "total_iter=" << total_iter << ", T=" << T << ", M=" << M << ", input_size=" << input_size;
  // figure out each multiplication sizes
  float scale_step = 1.2;
  for (int t = 0; t < T; ++t) {
    float scale_factor = pow(scale_step, -floor(T/2) + t);
    // output size of convolution
    int out_size = ceil( scale_factor * input_size ) + ( ksize - 1 ); 

    LOG(INFO) << "at " << t << " output size is " << out_size;
    
    Ns[t] = out_size * out_size ;
    total_N += Ns[t];
  }

  // Make matrices to be multiplied
  // W:
  shared_ptr<Blob<float> > W(new Blob<float>(M, channels, ksize, ksize));

  // individual ones:
  vector<shared_ptr<Blob<float> > > small_buffers( T );
  vector<shared_ptr<Blob<float> > > small_outputs( T );
  for (int t = 0; t < T; ++t ) {
    small_buffers[t].reset(new Blob<float>(1, K, Ns[t], 1, false));
    small_outputs[t].reset(new Blob<float>(1, M, Ns[t], 1));
  }
  // one big one
  shared_ptr<Blob<float> > big_buffer(new Blob<float>(1, K, total_N, 1, false) );
  // its output:
  shared_ptr<Blob<float> > big_output(new Blob<float>(1, M, total_N, 1) );


  const float* weight = W->gpu_data();
  // start comparison:
  LOG(INFO) << "*** Benchmark begins ***";
  Timer big_timer;
  big_timer.Start();
  // one big multiplication
  for (int i = 0; i < total_iter; ++i ) {
    float* out = big_output->mutable_gpu_data();
    float* data = big_buffer->mutable_gpu_data();  
    caffe_gpu_gemm<float>(CblasNoTrans, CblasNoTrans, M, total_N, K, 
			  1., weight, data, 0., out);
  }
  float total_big_time = big_timer.MilliSeconds();
  LOG(INFO) << "Big multiplicaion took " << total_big_time << " milli seconds, " 
	    << " average time is " << total_big_time / total_iter << " milli seconds";


  Timer small_timer; 
  small_timer.Start();
  // T many small ones:
  for (int i = 0; i < total_iter; ++i ) {
    for (int t = 0; t < T; ++t ) {
      float* out = small_outputs[t]->mutable_gpu_data();
      float* data = small_buffers[t]->mutable_gpu_data();
      caffe_gpu_gemm<float>(CblasNoTrans, CblasNoTrans, M, Ns[t], K, 
			    1., weight, data, 0., out);    
    }
  }
  float total_small_time = small_timer.MilliSeconds();
  LOG(INFO) << "Multiplication in Batches took " << total_small_time << " milli seconds."
	    << " average time is " << total_small_time / total_iter << " milli seconds";  

  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}

