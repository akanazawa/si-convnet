#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.
# then splits the training data into training (50,000) and validation (10,000)
# and computes mean image
#
# Angjoo Kanazawa

EXAMPLES=../../build/examples/lenet
TOOLS=../../build/tools
# DATA_SRC=/scratch0/data/mnist/
DATA_SRC=../../data/mnist/
DATA=../../data/mnist/

echo "Creating leveldb..."

rm -rf $DATA/mnist-train-leveldb $DATA/mnist-trainval-leveldb $DATA/mnist-val-leveldb
rm -rf $DATA/mnist-test-leveldb
rm -rf $DATA/mnist-trainval-image-mean

$EXAMPLES/convert_mnist_data.bin $DATA_SRC/train-images-idx3-ubyte $DATA_SRC/train-labels-idx1-ubyte $DATA/mnist-trainval-leveldb
$EXAMPLES/convert_mnist_data.bin $DATA_SRC/t10k-images-idx3-ubyte $DATA_SRC/t10k-labels-idx1-ubyte $DATA/mnist-test-leveldb

echo "Splitting into training and validation"

$TOOLS/split_training.bin $DATA/mnist-trainval-leveldb $DATA/mnist-train-leveldb $DATA/mnist-val-leveldb 10000

echo "Creating mean image using mnist-train-leveldb"

$TOOLS/compute_image_mean.bin $DATA/mnist-trainval-leveldb $DATA/mnist-trainval-image-mean

echo "Done."
