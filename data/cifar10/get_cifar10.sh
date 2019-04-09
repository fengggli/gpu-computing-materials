#!/usr/bin/env sh
# This scripts downloads the CIFAR10 (binary version) data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

echo "Downloading..."

wget -q --no-check-certificate http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

echo "Unzipping..."

tar -xf cifar-10-binary.tar.gz && rm -f cifar-10-binary.tar.gz
mv cifar-10-batches-bin/* . && rm -rf cifar-10-batches-bin

echo "concatenate all train set.."

cat data_batch_1.bin  data_batch_2.bin data_batch_3.bin data_batch_4.bin data_batch_5.bin > data_batch_all.bin

echo "Done."
