from .tensor import Tensor
import random
import math

# -----------------------------
# Linear Layer (Unchanged)
# -----------------------------
class Linear:
    def __init__(self, in_features, out_features):
        """
        Implements a fully connected layer: y = xW + b
        """
        self.W = Tensor([
            [random.uniform(-0.01, 0.01) for _ in range(out_features)]
            for _ in range(in_features)
        ], requires_grad=True)

        self.b = Tensor([[0.0] * out_features], requires_grad=True)

    def __call__(self, x):
        """
        Forward pass: xW + b
        Supports arbitrary batch dimensions.
        """
        orig_shape = x.shape()
        if len(orig_shape) > 2:
            # Flatten all but last dim
            batch_size = 1
            for dim in orig_shape[:-1]:
                batch_size *= dim
            reshaped = x.reshape((batch_size, orig_shape[-1]))
            out = reshaped.matmul(self.W) + self.b
            return out.reshape(orig_shape[:-1] + (self.W.shape()[1],))
        else:
            return x.matmul(self.W) + self.b

# -----------------------------
# Fixed LayerNorm (No NumPy, Fully Compatible)
# -----------------------------
class LayerNorm:
    def __init__(self, embed_dim, eps=1e-5):
        """
        Implements Layer Normalization over the last dimension.
        """
        self.eps = eps
        self.gamma = Tensor([[1.0] * embed_dim], requires_grad=True)
        self.beta = Tensor([[0.0] * embed_dim], requires_grad=True)

    def __call__(self, x):
        """
        x: Tensor of shape (batch, seq_len, embed_dim)
        returns: Tensor of same shape
        """
        shape = x.shape()
        if len(shape) != 3:
            raise ValueError(f"LayerNorm expects input of shape (batch, seq_len, embed_dim), got {shape}")

        batch_size, seq_len, dim = shape
        normalized = []

        for batch in x.data:
            norm_batch = []
            for vec in batch:
                mean = sum(vec) / dim
                var = sum((v - mean) ** 2 for v in vec) / dim
                norm = [(v - mean) / ((var + self.eps) ** 0.5) for v in vec]
                scaled = [norm[i] * self.gamma.data[0][i] + self.beta.data[0][i] for i in range(dim)]
                norm_batch.append(scaled)
            normalized.append(norm_batch)

        return Tensor(normalized, requires_grad=x.requires_grad)

