## Presentation 2

Sections are:

1. Introduce detailed convolution NN 
    * What are they used for?
    * why are they used ?
    * speed considerations
    * Layer construction
        * Common structures
        * weights replaced by "convolutions"
    * Pooling, padding, bias, other general concepts
    * backpropagation in convolution network

2. ResNet
    * Identiy layers should be able to be introduced with no negative consequences, but 
      this is not true in a nornal deep network. 
        * "Vanishing gradients" : Motivation for resnet (gradients get less effective as network 
          depth increases)      
    * Structure
        * Blocks 
        * Shortcut or feedforward
        * other - summary from the paper

3. Implementation plans 
    * Brief notes on other current implementations 
    * General layout of the software
        * gonna make a framework
        * with plug in activation function
        * how gradients plug in
    * Challenges we expect to face
    * GPU and other engineering considerations
        * potential optimizations relative to existing implementations
