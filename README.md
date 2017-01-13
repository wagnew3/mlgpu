##Algorithms in this library have not been extensively tested; some likely have bugs
**William Agnew's Machine Learning Library**

This library was implement largely for learning purposes; I do not have the time to implement the optimizations and features necessary to compete with the likes of TensorFlow or Theano.
However, implementing everything from scratch, including GPU acceleration, was a fun programming challenge that gave me a much deeper intuition for both machine learning algorithms and the structure of popular machine learning libraries. I talk about what I learned in the last section.

Many of the basics of reinforcement machine learning I learned from https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html, and supervised machine learning http://neuralnetworksanddeeplearning.com/

##Requirements
* JCUDA >=0.6 (and necessary GPU and CUDA setup)
* Apache Commons Math 3.x

##Easy Things to Run
* Reinforcement Learning+Neural Network State Valuation Checkers: src/checkers/testCheckers.java
<img src="https://github.com/wagnew3/mlgpu/blob/master/data/Checkers.jpg" width="400">
* MNIST: src/test/MNISTNumbers.java

##Reinforcement Learning
* Dynamic Programming Policy Iteration
* Dynamic Programming Value Iteration
* Off-Policy Monte Carlo
* Temporal Difference-Lambda
* Temporal Difference-Tesauro (as described in http://cling.csd.uwo.ca/cs346a/extra/tdgammon.pdf)

##Supervised Learning
* Feedforward Neural Networks
* Fully Connected Layers
* Convolution and Pooling Layers
* Fully Connected Layers Optimized for Sparse Inputs
* Convolution Layers Optimized for Sparse Inputs
* Sigmoid, TanH, ReLU, SoftMax
* Euclidean and Cross Entropy Loss
* L2 Regularization
* SGD, Multithreaded (Multiple GPU workers) SGD
* RProp, Multithreaded (Multiple GPU workers) RProp
* Nestrov Momentum
* Unsupervised Pretraining

##GPU Interface
* n-Dimensional FP32 GPU Accelerated Matrices with Wrappers to Many BLAS Calls (used JBLAS Library)
* n-Dimensional Sparse FP32 GPU Accelerated Matrices with Wrappers to Many BLAS Calls (used JBLAS Library)

##Lessons Learned
* Moving data between CPU and GPU memory is slow and must be essentially eliminated to not be a bottleneck.
* CUDA memory management is weird. When transferring a matrix from CPU to GPU memory, profiling revealed that it is faster to zero currently malloc'ed (but no longer needed) GPU memory and then copy the matrix from CPU memory into this zeroed GPU memory than to malloc new GPU memory and copy the matrix into that memory. However, I suspect there is much I do not understand about GPU memory (ex. asynchronous transfers).
* **Actually relevant to major ML libraries** Although I did not know it when I wrote this library, ML libraries like Theano and Tensorflow will copy over large chunks of memory to GPU-ex. a mini batch of training examples. This is a problem if the minibatch size exceeds the GPU memory size. While there are of course programmatic ways around this, my library completely avoids this problem by only copying matrices to GPU when a GPU function (ex. saxpy) has been called on them, and copying them back to CPU only when a CPU only function has been called on them. It would be a simple matter to combine this with making space if necessary on the GPU by sending matrices in GPU memory back to CPU memory, and drawing on the rich caching literature and predictable nature of most training algorithms to minimize the number of GPU-CPU matrix transfers while giving the ML training machine memory equal to CPU memory+GPU memory at no effort to the user-programmer.
* While I got away with not doing it for feedforward neural nets, it would be conceptually and programmatically easier to think of neural nets as graphs where edges represent data flow and vertices are differentiable functions: see my next machine learning library, https://github.com/wagnew/ComputeGraph, for implementation of this concept.