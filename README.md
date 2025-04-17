## What is dolphin?

It implements deep learning architecture and training logic without relying on NumPy, PyTorch, or any external libraries. Every operation—tensor arithmetic, backpropagation, attention, and optimization—is executed through hand-written, minimal Python logic.
The result is a transparent and flexible platform for understanding transformers, automatic differentiation, and the inner workings of modern ML, without hiding behind black-box libraries.

Dolphin is a minimalist computing substrate for symbolic tensor operations and transformer-based neural networks. It is:

A symbolic autodiff engine:
  - Includes a fully custom Tensor class with support for 1D, 2D, and 3D tensors
  - Tracks computation graphs and gradients
  - Implements backpropagation with minimal overhead

A pure Python transformer stack:
  - Multi-head self-attention
  - Layer normalization
  - GELU activations
  - Feedforward layers
  - Residual and normalization paths

A zero-dependency machine learning system:
  - No NumPy
  - No Torch
  - No Cython or compiled backend
  - Compatible with vanilla Python

---


Dolphin is built to help researchers, students, and curious developers explore how large language models and attention mechanisms actually function, without abstractions.

---

## Project Structure

The main Dolphin codebase is structured into several key modules:

### 1. `tensor.py`

The core of the symbolic engine.

- Defines the `Tensor` class.
- Supports 1D, 2D, and 3D nested lists as raw data.
- Includes autograd functionality: backward propagation, gradient accumulation, and dependency tracking.
- Implements all core operations:
  - Element-wise math: add, subtract, multiply, divide
  - Reductions: sum, mean, max, var
  - Matrix multiplication (`matmul`) for 2D and batched 3D inputs
  - Broadcasting support for scalar and tensor math
  - Reshaping, flattening, and transposition

### 2. `transformers.py`

Implements the full transformer encoder stack.

- `MultiHeadSelfAttention`: 
  - Implements scaled dot-product attention
  - Splits and recombines heads manually
  - Applies softmax and matmul using custom tensor logic

- `FeedForward`:
  - Two-layer MLP with GELU activation
  - Fully connected layers using internal `matmul` and bias addition

- `TransformerEncoderBlock`: 
  - Combines self-attention and feedforward blocks
  - Includes LayerNorm and residual connections

### 3. `activations.py`

Activation and loss functions.

- `softmax`: Written without NumPy. Normalizes logits across the last axis.
- `gelu`: Implements the Gaussian Error Linear Unit using a tanh approximation.
- `cross_entropy_loss`: Calculates negative log-likelihood between predictions and target labels.

### 4. `layers.py`

Layer utilities and normalization.

- `Linear`: Simple dense layer (y = xW + b), reshaped manually to support batch inputs.
- `LayerNorm`: Implements layer normalization without NumPy, applied over the last dimension.

### 5. `optimizers.py`

Optimization algorithms for parameter updates.

- `SGD`: Standard stochastic gradient descent.
- `Adam`: Adaptive Moment Estimation, written from scratch with moving averages and bias correction.
- `Momentum`: Implements SGD with momentum using manual velocity tracking.

---

## `DolphinTest01.py`

A complete example pipeline using Dolphin:

- Loads and tokenizes text from the Brown corpus using NLTK.
- Builds a vocabulary and transforms text into integer sequences.
- Initializes embedding weights manually.
- Feeds embedded tokens into a multi-layer transformer stack.
- Projects final token embeddings to a vocabulary-size output.
- Trains using backpropagation and Adam optimizer.
- Generates new sequences using a greedy sampling loop.

This file demonstrates that Dolphin can support:
- Text preprocessing
- Transformer-based sequence modeling
- End-to-end training without any library support
- Inference and autoregressive text generation

---

## Why Dolphin?

Dolphin was created to answer the question:  
**“What happens if we build a transformer from scratch, without any numerical libraries at all?”**

The result is not a high-performance framework. It is a clear, complete, and expressive computational substrate that exposes the full mechanics of deep learning.

It is ideal for:
- Teaching and learning how transformers and autograd work
- Experimenting with symbolic tensor operations
- Exploring what minimal, library-free ML looks like

---

## Requirements

Python 3.7+  
No external dependencies.

---

## License

Dolphin is released under the MIT License. You are free to use, modify, and distribute the code (see Citation file)

---

## Some Resources: 

- **Start with Tensors. This is the foundation for everything else**
  Think of it like a lego set. Tensors are your bricks, they are the basic building blocks of everything else. They handle data, operations, gradients, scalar differentiation, and form the foundation for learning. This is where you implement autodiff (automatic 
  differentiation) and backpropagation, which lets your model learn from mistakes. Then, computation graphs, they show how tensors are connected and how gradients flow backward through the network. This is especially important for Transformers, but that's the last step.
  
- **My build order**
  tensor engine (w/ autograd, etc) --> activations --> loss --> optimizers --> layers --> attention and transformers --> then training and sampling (aka, putting it all together).
  Will be making a better doc to explain each part, and how each module builds on one another. 
  
## For learning Transformers and attention
- **[Transformer using PyTorch | GeeksforGeeks](https://www.geeksforgeeks.org/transformer-using-pytorch/)**  
  A walkthrough of how to implement a Transformer using PyTorch components. Great for understanding how encoder/decoder blocks fit together.
- **[Attention is All You Need (paper)](https://arxiv.org/abs/1706.03762)**  
  The original paper that introduced the Transformer architecture. Dense but worth reading (line by line with code nearby). 
- **[Building a Transformer from Scratch: A Step-by-Step Guide](https://ebadsyed.medium.com/building-a-transformer-from-scratch-a-step-by-step-guide-d69cfa209ec3)**  
  A walkthrough of Transformer internals, broken down piece by piece, no hidden libraries. A high level explanation. 
- **[Python Lessons: Introduction to Transformers](https://pythonlessons.com/transformer-model/)**  
  Breakdown of the transformer model with annotated code and explanations.
- **[Transformers Explained](https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452/)**
  Explains how transformers work (visually). This is part 1, there are a couple different articles here that are really helpful. 

## For autodiff and Minimal Engines
- **[karpathy/micrograd](https://github.com/karpathy/micrograd)**  
  A tiny scalar-valued autograd engine and neural net library. Teaches how autodiff works w/ a PyTorch style API
- **[tinygrad/tinygrad](https://github.com/tinygrad/tinygrad)**  
  Like PyTorch but tiny (and readable). Implements both forward and backward passes, and can run on GPUs.
- **[rsokl/MyGrad](https://github.com/rsokl/MyGrad)**  
  Drop in automatic differentiation for NumPy. 

## Training GPTs and some Tensor building 
- **[karpathy/minGPT](https://github.com/karpathy/minGPT)**  
  A minimal PyTorch reimplementation of OpenAI’s GPT architecture.
- **[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)**  
  Simple repo to train/finetune GPT-style models, trains GPTs on single GPUs.
- **[easy-tensorflow/easy-tensorflow](https://github.com/easy-tensorflow/easy-tensorflow)**
  Explains TensorFlow components and how they work. 

## Neural Nets and Optimizers
- **[ShivamShrirao/dnn_from_scratch](https://github.com/ShivamShrirao/dnn_from_scratch)**  
  Implements CNNs, GANs, and more using only NumPy/CuPy, for low-level deep learning system design.
- **[Writing ML Optimizers from Scratch | Quassarian Viper](https://q-viper.github.io/2022/04/01/optimizers.html)**  
  A walkthrough of optimizers like SGD, Adam, RMSProp, all coded from scratch in Python.
- **[q-viper/ML-from-Basics](https://github.com/q-viper/ML-from-Basics)**  
  Minimal implementations of core ML algorithms like linear regression, SVMs, and more
- **[Deep Learning from Scratch | Seth Weidman](https://www.amazon.com/Deep-Learning-Scratch-Building-Principles/dp/9352139022)**  
  Book that builds neural networks from first principles in Python, theory and code.

This is a work in progress, and will be updated frequently :)
---

