# GPU-computing-materials

This is a collection of all materials of GPU computing course, which includes:
1. slides (or link to slides).
2. Meeting notes.
3. project code.

## Materials
1. Please refer to https://fengggli.github.io/dl-docs/ for learning materials for deep learning, I highly recommend the cs231 course from Stanford


## Presentation

#### presentation 1: intro to neural networks, deep learning, and backpropagation [2019-01-23]
* We had our first discussion on Jan 18, 2019, the outline of our first presentation is [here](/docs/presentation_1_outline.md)
* The slides are [here](https://docs.google.com/presentation/d/1mgcXAEhjIjccVH5eulKZUPSqueVNh7CkPg7BI5vt2kY/edit?usp=sharing)

#### presentation: 2 a deeper look at CNN's, ResNET, and implementation details:
* The second outline is [here](/docs/presentation_2_outline.md)
* The second presentation slides are [here]()

## Project
1. I will update my initial resnet implementation very soon. It will use numpy. 

#### Kernels
1. The first kernel we should have is  *an efficient convolution operation*. We can use profiling tools to compare the performance with existing libraries and show which optimization can be applied especially to Resnet nets.
2. Another focus is how to make the data copy from host to device less frequent. (How to represent the neuron layers in device memory instead)

