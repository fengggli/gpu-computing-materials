## Initial Plan for the project

#### Prepare
1. Read related work
2. Basic understanding of how popular DL frameworks handle this.

#### Steps

1. Test platform.

  * A working python version which can verify implementation of other optimized kernels.
  * Feng will show the NumPy version and its demo soon


2. Interface + Cython

  * A common Interface to kernels, so that each one of us can work independently to try different optimizations.
  * I will mostly like a C++ header. We can discuss it after the test platform is finalized.

3. Individual Kernel

  * Need present the sample code from Pytorch first, which is available from [here]()
  * Demonstrate our implementation
  * performance analysis
  

4. c++ ResNet platform

  * Finally most of the code will be in c++

5. Optimization
  * We need to compare with kernels implemented by vendor libraries.
  * Performance when trying different optimization methods.

#### Kernels
1. The first kernel we should have is  *an efficient convolution operation*. We can use profiling tools to compare the performance with existing libraries and show which optimization can be applied especially to Resnet nets.
2. Another focus is how to make the data copy from host to device less frequent. (How to represent the neuron layers in device memory instead)
