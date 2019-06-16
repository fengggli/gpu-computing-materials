# GPU-computing-materials

| branch | build status |
|--------|--------------|
| master | [![Build Master](https://travis-ci.com/fengggli/gpu-computing-materials.svg?token=21ngWpDjfcY4FxnxdNnA&branch=master)](https://travis-ci.com/fengggli/gpu-computing-materials) |
| feng | [![Build feng](https://travis-ci.com/fengggli/gpu-computing-materials.svg?token=21ngWpDjfcY4FxnxdNnA&branch=feng)](https://travis-ci.com/fengggli/gpu-computing-materials) |


* For the most recent documentations, see [here](https://fengggli.github.io/gpu-computing-materials)
* There is a changelog in the link above.

## Important: All the GPU-computing class related materials are in the [gpu-computing-submission branch](https://github.com/fengggli/gpu-computing-materials/tree/gpu-computing-submission)

* Because master is also for Feng's research use, we will trim some unused code in gpu-computing-submission branch before we merge it in.

## Code guideline

#### prepare (Optional)
  This is only needed for feng's experiments in CPU implementation(using cblas). you can ignore it.
sudo ./install-apt.sh

#### start

1. Prepare build directory

    ```
    git submodule update --init
    mkdir build
    cd build
    ```

2. configurate build options

    There are several backends for covolution, also we can choose FLT32/FLT64; you can check
    kicurrently enabled configurations in build/config.h

    * CI is build with minimal configurations
    ```
    cmake -DIS_CI_BUILD=on ..
    ```

    * to use CUDA with flt32
    ```
    cmake -DUSE_CUDA=on -DUSE_CUDNN=on -DAWNN_USE_FLT32=on -DCMAKE_BUILD_TYPE=Release  ..
    ```

    * To use NNPACK with flt32 use
    ```
    cmake -DUSE_NNPACK=on -DAWNN_USE_FLT32=on -DCMAKE_BUILD_TYPE=Release ..
    ```

3. In the builddir run:

    ```
    make
    ```

4. (optional )use cmake -DUSE_CLANG=on if you want to build with clang)

#### run test
1. run all tests(in the build directory)
```
make test
```
or
```
ctest
```

2. run individual test, e.g.
```
./tests/test_tensor
```
#### Documentation
See fengggli.github.io/gpu-computing-material
The sources of the doc in in docs/sources


This is a collection of all materials of GPU computing course, which includes:
1. slides (or link to slides).
2. Meeting notes.
3. Project code.

## Materials
1. Please refer to https://fengggli.github.io/dl-docs/ for learning materials for deep learning, I highly recommend the cs231 course from Stanford


## Presentations

#### presentation 1: intro to neural networks, deep learning, and backpropagation [2019-01-23]
* We had our first discussion on Jan 18, 2019, the outline of our first presentation is [here](/docs/presentation_1_outline.md)
* The slides are [here](https://docs.google.com/presentation/d/1mgcXAEhjIjccVH5eulKZUPSqueVNh7CkPg7BI5vt2kY/edit?usp=sharing)

#### presentation: 2 a deeper look at CNN's, ResNET, and implementation details:
* The second outline is [here](/docs/presentation_2_outline.md)
* The second presentation slides are [here](https://docs.google.com/presentation/d/1VNbwYfTrXLckYPZ6NOP41DlI_jujuesP1d6dcbzoBz4/edit?usp=sharing)

## Project

[Jan 26]: 
* We had a disussion on how we can start the project, the notes are [here](/docs/project_discuss_1.md)

[inital]:
* I will update my initial resnet implementation very soon. It will use numpy.

A implementation of a ANN framework for deep neural networks in CUDA
NOTE : this is using thrust with some custom wrapper classes. 
https://github.iu.edu/cmgoebel/504_C_GOEBEL_FINAL_PROJ/tree/master/CUDA_N_NET
