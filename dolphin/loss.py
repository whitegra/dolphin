from .tensor import Tensor
from .activations import softmax

def cross_entropy_loss(logits, targets):
    """
    Computes categorical cross-entropy loss using the custom Tensor class.
    Args:
        logits (Tensor): Raw model outputs before softmax.
        targets (Tensor): Target class indices as integers.
    Returns:
        Tensor: Scalar loss value with backprop enabled.
    """
    if logits is None or targets is None:
        raise ValueError("cross_entropy_loss received None for logits or targets.")

    probs = softmax(logits)
    log_probs = probs.log()

    # Flatten targets just in case they're in shape [[i], [j], ...]
    flat_targets = [int(t[0]) if isinstance(t, list) else int(t) for t in targets.data]

    selected_log_probs = []
    for i, target_index in enumerate(flat_targets):
        selected_row = log_probs.data[i]
        if target_index < 0 or target_index >= len(selected_row):
            raise IndexError(f"Target index {target_index} out of bounds for log_probs[{i}].")
        selected_log_probs.append([selected_row[target_index]])

    selected = Tensor(selected_log_probs, requires_grad=True)
    loss = -selected.mean()

    return loss

def binary_cross_entropy_loss(predictions, targets, eps=1e-9):
    """
    Computes Binary Cross-Entropy loss (for binary classification).
    Args:
        predictions (Tensor): Predicted probabilities (after sigmoid).
        targets (Tensor): Binary ground truth values (0 or 1).
    Returns:
        Tensor: Scalar loss.
    """
    # Clamp predictions to prevent log(0)
    preds_clamped = predictions.clip(eps, 1 - eps)
    targets_data = targets.data

    log_preds = preds_clamped.log()
    log_one_minus_preds = (Tensor([[1.0]]) - preds_clamped).log()
    one_minus_targets = Tensor([[1 - y for y in row] for row in targets_data])

    term1 = targets * log_preds
    term2 = one_minus_targets * log_one_minus_preds

    return -(term1 + term2).mean()
