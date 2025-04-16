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

Dolphin is released under the MIT License. You are free to use, modify, and distribute the code with proper attribution.

---
