# Archtecture-aware Neural Networks

A simple deep learning framework that optimizes task scheduling and memory usuage on different CPU/GPU architectures.

- I started as a research project trying to explore optimization on task scheduling and memory usuage with DNN workloads on different architecutures.
- The issues pages (some of those marked as "closed") contain partial preliminary results and interesting performance patterns.
- A classmate in GPU computing classes added GPU support too.

| branch | build status |
|--------|--------------|
| master | [![Build Master](https://travis-ci.com/fengggli/gpu-computing-materials.svg?token=21ngWpDjfcY4FxnxdNnA&branch=master)](https://travis-ci.com/fengggli/gpu-computing-materials) |
| feng | [![Build feng](https://travis-ci.com/fengggli/gpu-computing-materials.svg?token=21ngWpDjfcY4FxnxdNnA&branch=feng)](https://travis-ci.com/fengggli/gpu-computing-materials) |

## Instructions 

* For the most recent documentations online, see [here](https://fengggli.github.io/gpu-computing-materials)
* There is a changelog in the link above.
* To test with caffe, see [my forked caffe](https://github.com/fengggli/caffe/blob/fengggli-archlinux-cpuonly/models/resnet_simple/readme.md) 

#### prepare
```
git submodule update --init
mkdir build
cd build
```

#### Build
We use mkl for cpu gemm().
```
source /opt/intel/bin/compilervars.sh intel64
source /opt/intel/mkl/bin/mklvars.sh intel64
```

Then build with
```
cmake -DUSE_MKL=on -DAWNN_USE_FLT32=on ..
```

* BUILD in stampede2

in the builddir, run 
```
../scripts/build_stampede2.sh
```

When mkl is not avaible install openblas and build with -DUSE_OPENBLAS=on
```
sudo ./install-apt.sh
cmake -DUSE_OPENBLAS=on -DAWNN_USE_FLT32=on ..
```
