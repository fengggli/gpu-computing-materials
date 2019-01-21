# GPU-computing-materials

This is a collection of all materials of GPU computing course, which includes:
1. slides (or link to slides).
2. Meeting notes.
3. project code.

## Materials
1. Please refer to https://fengggli.github.io/dl-docs/ for learning materials for deep learning, I highly recommend the cs231 course from Stanford


## Presentation

#### presentation 1: intro to deep learning and neuron networks [2019-01-20]
* We had our first discussion on Jan 18, 2019, the outline of our first presentation is in [here](https://docs.google.com/document/d/1Q0GslX0j1lE9mc5EGhLfQAQOoWzfFo_pglrfK4IG5Lk/edit?usp=sharing)
* The slides is [here](https://docs.google.com/presentation/d/1mgcXAEhjIjccVH5eulKZUPSqueVNh7CkPg7BI5vt2kY/edit?usp=sharing)

#### presentation 2:

## Project
1. I will update my initial resnet implementation very soon. It will use numpy. 

#### Kernels
1. The first kernel we should have is  *an efficient convolution operation*. We can use profiling tools to compare the performance with existing libraries and show which optimization can be applied especially to Resnet nets.
2. Another focus is how to make the data copy from host to device less frequent. (How to represent the neuron layers in device memory instead)

