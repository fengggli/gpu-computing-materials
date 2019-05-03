# Seond meeting

March 6

####

Yuankuan will read the convolution and implement it in cuda
i and crhis will decide the interface and decide the testing code(can be ported from numpy)
chris will also look into the numpy code

Data?


#### Go through the numpy version

Materials:

1. Intel-Caffe, Optimize caffe in Intel Architecture, [this blog](https://software.intel.com/en-us/articles/caffe-optimized-for-intel-architecture-applying-modern-code-techniques), tells how to find performance hotspot and how to optimize it.
2. Performance of convnet, ultimately we can evaluate one full forward and backward , like [this page from soumith](https://github.com/soumith/convnet-benchmarks)
3. Caffe githubpage
  * It's clearly written in c++
  * nice test suite for nearly every components.
  * both cpp and cu
  * utilities we can reuse.


#### TODO

(I and II will be done in the same time)

I. Write kernels, using same signatures like those in caffe. Suppose you can reuse the datatype (low level)

  * Convolution, pooling. (im2col)
  * verify using the utils in caffe
  * statics (what's the performance of the kernel, gflops and membandwidth)

II . Understand the performance (high level)
Use caffe to reproduce the performance analytics in both GPU and CPU

  * time for a forward-backward
  * Memory usage
  * Data reuse
  * Hotspot for cpu/cuda time
  * NVprofiler and intel vtune

III. Integrate together TO CPU code, and experiments

  * simple fully connnect net -> shadow convnet -> add shortcut-> deep net.
  * We can reuse batchnorm and optimizer from caffe code.
  * Than we define the resnet in caffee

Hope we can finish all the code at the begining or may.


Maybe
  * I will have my own c++ code we can plug all the kernels in.

#### Minestones

TO be filled


