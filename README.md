[![Build Master](https://travis-ci.com/fengggli/gpu-computing-materials.svg?token=21ngWpDjfcY4FxnxdNnA&branch=master)](https://travis-ci.com/fengggli/gpu-computing-materials)
[![Build feng](https://travis-ci.com/fengggli/gpu-computing-materials.svg?token=21ngWpDjfcY4FxnxdNnA&branch=feng)](https://travis-ci.com/fengggli/gpu-computing-materials)


# Code guideline

#### prepare
Need cblas
sudo ./install-apt.sh

#### start
```
git submodule update --init
mkdir build
cd build
cmake .. 
  * use cmake -DUSE_CLANG=on if you want to build with clang)
make
```

# GPU-computing-materials

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
