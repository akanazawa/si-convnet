#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.
# then splits the training data into training (50,000) and validation (10,000)
# and computes mean image
#
# Angjoo Kanazawa

EXAMPLES=../../build/examples/cifar
TOOLS=../../build/tools
DATA_SRC=../../data/cifar10/
DATA=../../data/cifar10/

echo "Creating leveldb..."

rm -rf $DATA/*leveldb
rm -rf $DATA/cifar10-trainval-image-mean

$EXAMPLES/convert_cifar_data.bin $DATA_SRC $DATA

echo "Creating mean image using cifar-trainval-leveldb"

$TOOLS/compute_image_mean.bin $DATA/cifar-trainval-leveldb $DATA/cifar-trainval-image-mean

echo "Done."
