## Presentation 2

Sections are:

1. Introduce detailed convolution NN  (mainly by Chris)
    * What are they used for?
    * Why are they used ?
        * speed considerations
        * size 
    * Layer construction (christ intro, yuankun will explain more details)
        * Common structures
        * weights replaced by "convolutions"
    * Pooling, padding, bias, other general concepts (Yuankun)
    * back-propagation in convolution network (Christ, maybe)

2. ResNet (Mainly by Yuankun)

    * "Vanishing gradients" : Motivation for resnet (gradients get less effective as network 
       depth increases) (Feng)    
    * Identity layers should be able to be introduced with no negative consequences, but 
      this is not true in a nornal deep network. 
    * Structure Yuankun
        * Blocks 
        * Shortcut or feedforward
        * other - summary from the paper

3. Implementation plans (Feng) 
    * Brief notes on other current implementations 
    * General layout of the software
        * gonna make a framework
        * with plug in activation function
        * how gradients plug in
    * Challenges we expect to face
    * GPU and other engineering considerations
        * potential optimizations relative to existing implementations
