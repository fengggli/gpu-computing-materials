#!/usr/bin/env bash
# Run this in the build directory
cmake -DUSE_MKL=on -DAWNN_USE_FLT32=on -DUSE_ICC=on -DUSE_AVX512=on -DCMAKE_BUILD_TYPE=Release ..
