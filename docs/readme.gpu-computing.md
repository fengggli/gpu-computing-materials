Instructions for GPU-computing class

#### platform

sievert.cs.iupui.edu(with ubuntu 16.04 and cuda 10.0)
If cuda 10.0 is not detected when you login, check your login message and follow
instructions.

#### prepare
We use openblas for cpu gemm.
(In sievert you don't need this, since openblas is already installed)

```
sudo ./install-apt.sh
```

If you use another machine, make sure openblas library 
and headers are in searchable paths

#### start
```
git submodule update --init
mkdir build
cd build
cmake -DUSE_CUDA=on -DUSE_CUDNN=on -DAWNN_USE_FLT32=on -DCMAKE_BUILD_TYPE=Release .. 
make -j 16
```

#### run test
1. **In the build directory**, run unit tests(check the correctness of all our gpu kernels)
    * test cudnn convolution
    ```
    ./tests/test-layer-conv-cudnn
    ```
    
    * test our device convolution
    ```
    ./tests/test-layer-conv-device
    ```
    
    * test residual block using our optimized gpu kernels
    ```
    ./tests/test-resblock-device

    ```

2. **In the build directory**, run benchmarks(speed test for comparison between cudnn and our device code)

    * bench cudnn convolution
    ```
    ./bench/bench-conv-cudnn
    ```
    
    * bench our optimized convolution
    ```
    ./bench/bench-conv-device
    ```

#### Documentation
* [Final presentation Slides](/docs/Project%20Final%20Presentation.pdf)
* [Final Report](/docs/gpu-computing-19/gpu_resnet_project_report.pdf)

* TODO: report 

* Other supporting materials are in in docs/sources

* Please refer to https://fengggli.github.io/dl-docs/ for learning materials for deep learning, I highly recommend the cs231 course from Stanford

