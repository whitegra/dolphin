import math
from .tensor import Tensor

def softmax(x: Tensor) -> Tensor:
    """
    Softmax along the last axis (for attention scores or logits).
    Supports 2D or 3D tensors.
    """
    shape = x.shape()
    output = []

    if len(shape) == 3:
        for mat in x.data:  # shape: (batch, seq_len, seq_len)
            mat_out = []
            for row in mat:
                max_val = max(row)
                exps = [math.exp(v - max_val) for v in row]
                total = sum(exps)
                mat_out.append([v / total for v in exps])
            output.append(mat_out)

    elif len(shape) == 2:
        for row in x.data:
            max_val = max(row)
            exps = [math.exp(v - max_val) for v in row]
            total = sum(exps)
            output.append([v / total for v in exps])

    else:
        raise ValueError(f"Softmax only supports 2D or 3D tensors, got {shape}")

    return Tensor(output, requires_grad=x.requires_grad)


def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Computes cross entropy between predictions and true labels.
    logits: (batch, vocab_size)
    targets: (batch, 1) or (batch,)
    """
    probs = softmax(logits)
    log_probs = probs.log()

    if isinstance(targets.data[0], list):
        target_indices = [i[0] for i in targets.data]
    else:
        target_indices = [i for i in targets.data]

    loss_values = []
    for i, idx in enumerate(target_indices):
        prob = probs.data[i][idx]
        loss_values.append(-math.log(prob + 1e-9))  # avoid log(0)

    avg_loss = sum(loss_values) / len(loss_values)
    return Tensor([[avg_loss]], requires_grad=True)


def gelu(x: Tensor) -> Tensor:
    """
    GELU activation using the tanh approximation.
    Applies element-wise.
    """
    def gelu_scalar(val):
        return 0.5 * val * (1 + math.tanh(math.sqrt(2 / math.pi) * (val + 0.044715 * val**3)))

    if len(x.shape()) == 2:
        out_data = [[gelu_scalar(v) for v in row] for row in x.data]
    else:
        out_data = [[[gelu_scalar(v) for v in row] for row in mat] for mat in x.data]

    return Tensor(out_data, requires_grad=x.requires_grad)
