Locally Scale-Invariant ConvNet Caffe Implementation
=============

This packages implements the scale-invariant ConvNet used in our [NIPS
2014 Deep Learning & Representation Workshop paper](http://www.umiacs.umd.edu/~kanazawa/papers/sicnn_workshop2014.pdf).

It's based on [BVLC's Caffe](http://caffe.berkeleyvision.org), final merge
with BVLC/master was on [Oct 20th 2014](https://github.com/BVLC/caffe/commit/c18d22eb92488f02c0256a3fe4ac20a8ad827596).

Installation
---
Requires all of Caffe's prerequisite packages. Compile as you would
compile Caffe i.e. have the right Makefile.config and
```
make all
make test
make runtest
```

Changes to BVLC/Caffe
---
The major additions are:

1. `util/transformation.(hpp/cpp/cu)`
   Misc functions needed to apply image transformation using NN
   or bilinear interpolation.
2. `ticonv_layer.cpp`
   `TIConvolutionLayer` a wrapper around `UpsamplingLayer`,
   `tiedconv_layer` and `DownpoolLayer`. This is what you can use
   instead of convolution layer to use SI-Conv layer.
3. `up_layer.cpp`
   Contains `UpsamplingLayer` which applies user specified interpolations to the
   bottom blob. i.e. TransformationLayer.
4. `downpool_layer.cpp`
   Contains `DownpoolLayer`, which is almost the same as `UpsamplingLayer`, but after applying transformations, crops
   the inputs into a canonical shape and does max-pooling over all transformations. 
5. `tiedconv_layer.cpp`
   Convolutional layer that can apply convolution to multiple inputs
   using the same weight. Very close to current (Jan 2015) Caffe's
   `ConvolutionalLayer` except that the input size can vary.
6. `util/imshow.(hpp/cpp)`
   (not necessary), used for debugging images in C++ using openCV
   behaves like matlab's imshow and montage.
7. And all the misc changes needed to adapt the changes into the rest of the
   code.

All major changes are implemented in both CPU and GPU with tests.

Technical Note: since CUDA's `atomicAdd`, required in backprop fo transformation
layer isn't available for doubles, this code only runs for `float`
instantiation of Caffe (which shouldn't be a problem since default
Caffe runs in `float`). But because of that, all explicit instantiation
for `doubles` are commented out.


How to use SI-Conv Layer instead of Conv Layer
---
In your protofiles, replace the type of the layer from `CONVOLUTION` to
`TICONV` and add transformations that you want to apply to this
layer. Note that `TICONV` layer assumes that the first transformation is always
identity and is the canonical size.

Example:

A Convolution Layer:

	 layers {
	   name: "conv1"
	   type: CONVOLUTION
	   bottom: "data"
	   top: "conv1"
	   blobs_lr: 1.
	   blobs_lr: 2.
	   weight_decay: 1.
	   weight_decay: 0.
	   convolution_param {
		 num_output: 36
		 kernel_size: 7
		 stride: 1
		 weight_filler {
		   type: "gaussian"
		   std: 0.01
		 }
		 bias_filler {
		   type: "constant"
		 }
	   }
	 }

A Scale-Invariant Convolution Layer:

	 layers {
	   name: "conv1"
	   type: CONVOLUTION
	   bottom: "data"
	   top: "conv1"
	   blobs_lr: 1.
	   blobs_lr: 2.
	   weight_decay: 1.
	   weight_decay: 0.
	   convolution_param {
		 num_output: 36
		 kernel_size: 7
		 stride: 1
		 weight_filler {
		   type: "gaussian"
		   std: 0.01
		 }
		 bias_filler {
		   type: "constant"
		 }		 
	   }
	   transformations {}
	   transformations { scale: 0.63 }
	   transformations { scale: 0.7937 }
	   transformations { scale: 1.2599 }
	   transformations { scale: 1.5874 }
	   transformations { scale: 2 }
	 }

Transformations parameter accepts parameters:
- `scale`: scale-factor
- `rotation`: rotation in degrees
- `border`: border option similar to matlab {0=crop (default), 1=clamp, 2=reflect} 
- `interp`: interpolation option {0=Nearest Neighbor, 1=Bilinear
  (default)}
So it can handle transformations other than scale as well.
Sample protos can be found in `models/sicnn/protos`.

Replicating the results on paper
---
Get the MNIST-Scale train/test folds in hdf5 format (mean subtracted) from
[here](http://angjookanazawa.com/sicnn/mnist-sc-table1.tar.gz)
and unzip it in `data/mnist` or from this directory:

```
cd data/mnist
wget http://angjookanazawa.com/sicnn/mnist-sc-table1.tar.gz
tar vxzf mnist-sc-table1.tar.gz
```

`models/sicnn` has sample prototxt for vanila convnet, hierarchical
convnet of Farabet et al [1] and si-convnet used in ther paper for
split 1. From this directory each one can be run with:

```
./train_all.sh cnn
./train_all.sh farabet
./train_all.sh sicnn
```

Note: There was a minor bug in the transformation code which further
improved SI-ConvNet mean error on the 6 train/test fold from 3.13%
to 2.93%. The performance on the other two models stayed the same.
On this split 1, this SI-ConvNet should get something like 2.91% error.

Citing
---
If you find any part of this code useful, please consider
citing:

	@misc{kanazawa14,
	author    = {Angjoo Kanazawa and Abhishek Sharma and David W. Jacobs},
	title     = {Locally Scale-Invariant Convolutional Neural Networks},
	year      = {2014},
	url       = {http://arxiv.org/abs/1412.5104},
	Eprint = {arXiv:1412.5104}
	}

as well as the Caffe Library.

	@misc{Jia13caffe,
		Author = {Yangqing Jia},
		Title = { {Caffe}: An Open Source Convolutional Architecture
		for Fast Feature Embedding},
		Year = {2013},
		Howpublished = {\url{http://caffe.berkeleyvision.org/}}
	}

Questions, comments, bug report
---
Please direct any questions, comment, bug report etc to
kanazawa[at]umiacs[dot]umd[dot]edu.

[1] Clement Farabet, Camille Couprie, Laurent Najman and Yann LeCun,
"Learning Hierarchical Features for Scene Labeling", IEEE PAMI 2013.
