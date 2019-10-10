# Run this in the build directory
cmake .. \
  -DUSE_MKL=on \
  -DAWNN_USE_FLT32=on \
  -DUSE_AVX512=on \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
  #-DCMAKE_BUILD_TYPE=Release

  #-DUSE_ICC=on \  icc doesn't give performance improve 
  #-DUSE_ICC=on -DUSE_AVX512=on  \
  #-DMKL_ROOT="${HOME}/Workspace/awnn/caffe/external/mkl/mklml" \
