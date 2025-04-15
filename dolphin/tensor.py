import math
import random

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = self._validate_and_wrap(data)
        self.requires_grad = requires_grad
        self.grad = self._zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set()

    def _validate_and_wrap(self, data):
        if isinstance(data, (int, float)):
            return [[data]]
        elif isinstance(data, list):
            if all(isinstance(row, list) for row in data):
                if all(isinstance(x, (int, float)) for row in data for x in row):
                    return data  # 2D
                elif all(isinstance(x, list) for row in data for x in row):
                    return data  # 3D
            elif all(isinstance(x, (int, float)) for x in data):
                return [data]
        raise TypeError("Tensor data must be scalar, 1D, 2D, or 3D nested lists.")

    def _zeros_like(self, data):
        if isinstance(data[0], list) and isinstance(data[0][0], list):  # 3D
            return [[[0 for _ in row] for row in mat] for mat in data]
        return [[0 for _ in row] for row in data]

    def shape(self):
        if isinstance(self.data[0], list) and isinstance(self.data[0][0], list):
            return (len(self.data), len(self.data[0]), len(self.data[0][0]))
        return (len(self.data), len(self.data[0]))

    def __repr__(self):
        return f"Tensor(shape={self.shape()}, requires_grad={self.requires_grad})"

    def _ensure_tensor(self, other):
        if not isinstance(other, Tensor):
            shape = self.shape()
            if len(shape) == 3:
                return Tensor([[[other for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])])
            elif len(shape) == 2:
                return Tensor([[other for _ in range(shape[1])] for _ in range(shape[0])])
        return other

    def __add__(self, other):
        other = self._ensure_tensor(other)

        def add(a, b):
            if isinstance(a, list):
                return [add(ai, bi) for ai, bi in zip(a, b)]
            return a + b

        result = add(self.data, other.data)
        out = Tensor(result, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad)
            if other.requires_grad:
                other._accumulate_grad(out.grad)

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            def scalar_mul(x): return [scalar_mul(i) if isinstance(i, list) else i * other for i in x]
            return Tensor(scalar_mul(self.data), requires_grad=self.requires_grad)

        other = self._ensure_tensor(other)

        def mul(a, b):
            if isinstance(a, list):
                return [mul(ai, bi) for ai, bi in zip(a, b)]
            return a * b

        result = mul(self.data, other.data)
        out = Tensor(result, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(mul(out.grad, other.data))
            if other.requires_grad:
                other._accumulate_grad(mul(out.grad, self.data))

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __truediv__(self, other):
        other = self._ensure_tensor(other)

        def div(a, b):
            if isinstance(a, list):
                return [div(ai, bi) for ai, bi in zip(a, b)]
            return a / b

        result = div(self.data, other.data)
        out = Tensor(result, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(div(out.grad, other.data))
            if other.requires_grad:
                neg_grad = []
                for gr, a, b in zip(out.grad, self.data, other.data):
                    row = [-g * a / (b ** 2) for g, a, b in zip(gr, a, b)]
                    neg_grad.append(row)
                other._accumulate_grad(neg_grad)

        out._backward = _backward
        out._prev = {self, other}
        return out

    def sum(self, axis=None, keepdims=False):
        shape = self.shape()
        data = self.data

        if axis is None:
            total = sum(self._flatten(data))
            return Tensor([[total]] if keepdims else total)

        if axis == -1:
            axis = len(shape) - 1

        if len(shape) == 2:
            if axis == 0:
                cols = [sum(col) for col in zip(*data)]
                return Tensor([cols] if keepdims else cols)
            elif axis == 1:
                rows = [[sum(row)] if keepdims else sum(row) for row in data]
                return Tensor(rows)

        elif len(shape) == 3 and axis == 2:
            result = []
            for mat in data:
                mat_sum = [[sum(row)] if keepdims else sum(row) for row in mat]
                result.append(mat_sum)
            return Tensor(result)

        raise NotImplementedError("Sum only supports axis=-1 or last dim for now.")

    def max(self, axis=None, keepdims=False):
        shape = self.shape()
        data = self.data

        if axis is None:
            val = max(self._flatten(data))
            return Tensor([[val]] if keepdims else val)

        if axis == -1:
            axis = len(shape) - 1

        if len(shape) == 2:
            if axis == 0:
                cols = [max(col) for col in zip(*data)]
                return Tensor([cols] if keepdims else cols)
            elif axis == 1:
                rows = [[max(row)] if keepdims else max(row) for row in data]
                return Tensor(rows)

        elif len(shape) == 3 and axis == 2:
            result = []
            for mat in data:
                mat_max = [[max(row)] if keepdims else max(row) for row in mat]
                result.append(mat_max)
            return Tensor(result)

        raise NotImplementedError("Max only supports axis=-1 or last dim for now.")

    def exp(self):
        def apply(x):
            return [apply(i) if isinstance(i, list) else math.exp(i) for i in x]
        result = apply(self.data)
        return Tensor(result, requires_grad=self.requires_grad)

    def log(self):
        def apply(x):
            return [apply(i) if isinstance(i, list) else math.log(i + 1e-9) for i in x]
        result = apply(self.data)
        return Tensor(result, requires_grad=self.requires_grad)
    
    def sqrt(self):
        def apply_sqrt(data):
            if isinstance(data, list):
                return [apply_sqrt(x) for x in data]
            return math.sqrt(data)
    
        result = apply_sqrt(self.data)
        out = Tensor(result, requires_grad=self.requires_grad)
    
        def _backward():
            if self.requires_grad:
                def grad_sqrt(orig, grad_out):
                    if isinstance(orig, list):
                        return [grad_sqrt(o, g) for o, g in zip(orig, grad_out)]
                    return grad_out / (2 * math.sqrt(orig + 1e-9))  # add eps to prevent div by 0
    
                grad = grad_sqrt(self.data, out.grad)
                self._accumulate_grad(grad)
    
        out._backward = _backward
        out._prev = {self}
        return out
    
        
    def mean(self, axis=None, keepdims=False):
        shape = self.shape()
    
        if axis is None:
            flat = self._flatten(self.data)
            return Tensor([[sum(flat) / len(flat)]], requires_grad=self.requires_grad)
    
        if axis == -1:
            axis = len(shape) - 1
    
        if len(shape) == 2:
            if axis == 0:
                result = [sum(col) / len(col) for col in zip(*self.data)]
                return Tensor([result] if keepdims else result, requires_grad=self.requires_grad)
            elif axis == 1:
                result = [[sum(row) / len(row)] if keepdims else sum(row) / len(row) for row in self.data]
                return Tensor(result, requires_grad=self.requires_grad)
    
        elif len(shape) == 3:
            if axis == 2:
                result = []
                for mat in self.data:
                    mat_means = []
                    for row in mat:
                        mean_val = sum(row) / len(row)
                        mat_means.append([mean_val] if keepdims else mean_val)
                    result.append(mat_means)
                return Tensor(result, requires_grad=self.requires_grad)
    
        raise NotImplementedError(f"mean(axis={axis}) not implemented for shape {shape}")
   
    def var(self, axis=None, keepdims=False):
        shape = self.shape()
    
        if axis is None:
            flat = self._flatten(self.data)
            mean = sum(flat) / len(flat)
            variance = sum((x - mean) ** 2 for x in flat) / len(flat)
            return Tensor([[variance]] if keepdims else variance, requires_grad=self.requires_grad)
    
        if axis == -1:
            axis = len(shape) - 1
    
        if len(shape) == 2:
            if axis == 0:
                cols = list(zip(*self.data))
                variances = []
                for col in cols:
                    m = sum(col) / len(col)
                    v = sum((x - m) ** 2 for x in col) / len(col)
                    variances.append(v)
                return Tensor([variances] if keepdims else variances, requires_grad=self.requires_grad)
    
            elif axis == 1:
                variances = []
                for row in self.data:
                    m = sum(row) / len(row)
                    v = sum((x - m) ** 2 for x in row) / len(row)
                    variances.append([v] if keepdims else v)
                return Tensor(variances, requires_grad=self.requires_grad)
    
        elif len(shape) == 3 and axis == 2:
            result = []
            for mat in self.data:
                mat_vars = []
                for row in mat:
                    m = sum(row) / len(row)
                    v = sum((x - m) ** 2 for x in row) / len(row)
                    mat_vars.append([v] if keepdims else v)
                result.append(mat_vars)
            return Tensor(result, requires_grad=self.requires_grad)
    
        raise NotImplementedError(f"var(axis={axis}) not implemented for shape {shape}")
    

    def transpose(self):
        shape = self.shape()
        if len(shape) == 2:
            transposed = list(map(list, zip(*self.data)))
            return Tensor(transposed, requires_grad=self.requires_grad)
        elif len(shape) == 3:
            transposed = []
            for mat in self.data:
                transposed.append([list(col) for col in zip(*mat)])
            return Tensor(transposed, requires_grad=self.requires_grad)
        else:
            raise NotImplementedError("Transpose only supports 2D or 3D tensors.")
        return Tensor(transposed, requires_grad=self.requires_grad)
    
    def reshape(self, new_shape):
        flat = self._flatten(self.data)
    
        def build(shape, flat_iter):
            if len(shape) == 1:
                return [next(flat_iter) for _ in range(shape[0])]
            return [build(shape[1:], flat_iter) for _ in range(shape[0])]
    
        # Handle -1 (infer dimension)
        if -1 in new_shape:
            known = 1
            for dim in new_shape:
                if dim != -1:
                    known *= dim
            inferred = len(flat) // known
            new_shape = tuple(inferred if dim == -1 else dim for dim in new_shape)
    
        flat_iter = iter(flat)
        reshaped = build(new_shape, flat_iter)
    
        return Tensor(reshaped, requires_grad=self.requires_grad)

    def matmul(self, other):
        other = self._ensure_tensor(other)
        a_shape, b_shape = self.shape(), other.shape()

        if len(a_shape) == 2 and len(b_shape) == 2:
            result = [[sum(self.data[i][k] * other.data[k][j] for k in range(a_shape[1]))
                       for j in range(b_shape[1])]
                       for i in range(a_shape[0])]
            
        elif len(a_shape) == 3 and len(b_shape) == 2:
            if a_shape[2] != b_shape[0]:
                raise ValueError(f"Incompatible shapes for matmul: {a_shape} @ {b_shape}")
        
            result = []
            for mat in self.data:  # shape: (seq_len, embed_dim)
                mat_result = []
                for row in mat:
                    row_result = []
                    for j in range(b_shape[1]):
                        dot = sum(row[k] * other.data[k][j] for k in range(b_shape[0]))
                        row_result.append(dot)
                    mat_result.append(row_result)
                result.append(mat_result)

        elif len(a_shape) == 3 and len(b_shape) == 3:
            if a_shape[0] != b_shape[0] or a_shape[2] != b_shape[1]:
                raise ValueError(f"Incompatible shapes for batched matmul: {a_shape} @ {b_shape}")
            result = [[[sum(self.data[b][i][k] * other.data[b][k][j] for k in range(a_shape[2]))
                        for j in range(b_shape[2])]
                        for i in range(a_shape[1])]
                        for b in range(a_shape[0])]
        # Handle (batch, 1, embed_dim) @ (embed_dim, out_dim)
        elif len(a_shape) == 3 and len(b_shape) == 2 and a_shape[1] == 1 and a_shape[2] == b_shape[0]:
            result = []
            for mat in self.data:  # mat: [[vec]]
                row_result = []
                for j in range(b_shape[1]):
                    dot = sum(mat[0][k] * other.data[k][j] for k in range(b_shape[0]))
                    row_result.append(dot)
                result.append([row_result])  # keep 3D
            return Tensor(result, requires_grad=self.requires_grad or other.requires_grad)
        else:
            raise NotImplementedError("Unsupported shapes for matmul.")

        return Tensor(result, requires_grad=self.requires_grad or other.requires_grad)

    def _flatten(self, data):
        if isinstance(data, list):
            return [x for sub in data for x in self._flatten(sub)]
        return [data]

    def _accumulate_grad(self, grad):
        def add(a, b):
            if isinstance(a, list):
                return [add(x, y) for x, y in zip(a, b)]
            return a + b
        self.grad = add(self.grad, grad)

    def backward(self):
        if self.grad is None:
            self.grad = self._zeros_like(self.data)
            for i in range(len(self.grad)):
                for j in range(len(self.grad[0])):
                    self.grad[i][j] = 1

        visited = set()
        topo = []

        def build(t):
            if t not in visited:
                visited.add(t)
                for p in t._prev:
                    build(p)
                topo.append(t)

        build(self)
        for t in reversed(topo):
            t._backward()

    @staticmethod
    def zeros(shape):
        if len(shape) == 2:
            return Tensor([[0 for _ in range(shape[1])] for _ in range(shape[0])])
        elif len(shape) == 3:
            return Tensor([[[0 for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])])
        else:
            raise ValueError("Invalid shape for Tensor.zeros")

    @staticmethod
    def ones(shape):
        if len(shape) == 2:
            return Tensor([[1 for _ in range(shape[1])] for _ in range(shape[0])])
        elif len(shape) == 3:
            return Tensor([[[1 for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])])
        else:
            raise ValueError("Invalid shape for Tensor.ones")

    @staticmethod
    def rand(shape):
        if len(shape) == 2:
            return Tensor([[random.random() for _ in range(shape[1])] for _ in range(shape[0])])
        elif len(shape) == 3:
            return Tensor([[[random.random() for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])])
        else:
            raise ValueError("Invalid shape for Tensor.rand")
