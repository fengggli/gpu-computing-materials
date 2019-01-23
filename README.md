# GPU-computing-materials

This is a collection of all materials of GPU computing course, which includes:
1. slides (or link to slides).
2. Meeting notes.
3. project code.

## Materials
1. Please refer to https://fengggli.github.io/dl-docs/ for learning materials for deep learning, I highly recommend the cs231 course from Stanford


## Presentation

#### presentation 1: intro to deep learning and neuron networks [2019-01-20]
* We had our first discussion on Jan 18, 2019, the outline of our first presentation (also a little bit about the second one) is in [here](/docs/meeting-note-Jan-20.md)
* The slides is [here](https://docs.google.com/presentation/d/1mgcXAEhjIjccVH5eulKZUPSqueVNh7CkPg7BI5vt2kY/edit?usp=sharing)

#### presentation 2:
* The second outline will be [here](/docs/meeting-note-Jan-20.md)

#### presentation 3:
* implementation strategy,
* backpropagation in convolution network

## Project
1. I will update my initial resnet implementation very soon. It will use numpy. 

#### Kernels
1. The first kernel we should have is  *an efficient convolution operation*. We can use profiling tools to compare the performance with existing libraries and show which optimization can be applied especially to Resnet nets.
2. Another focus is how to make the data copy from host to device less frequent. (How to represent the neuron layers in device memory instead)

