from .tensor import Tensor
from .activations import softmax, gelu
from .layers import LayerNorm
import math
import random


class MultiHeadSelfAttention:
    def __init__(self, embed_dim, num_heads):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        def init_weight():
            return Tensor([
                [(random.random() * 0.02 - 0.01) for _ in range(embed_dim)]
                for _ in range(embed_dim)
            ], requires_grad=True)

        self.W_q = init_weight()
        self.W_k = init_weight()
        self.W_v = init_weight()
        self.W_o = init_weight()

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape()
        heads = []
        for b in range(batch_size):
            for h in range(self.num_heads):
                head = []
                for s in range(seq_len):
                    start = h * self.head_dim
                    end = (h + 1) * self.head_dim
                    head.append(x.data[b][s][start:end])
                heads.append(head)
        return Tensor(heads, requires_grad=x.requires_grad)

    def combine_heads(self, x, batch_size):
        num_heads = self.num_heads
        seq_len = len(x.data[0])
        combined = []
        for b in range(batch_size):
            rows = []
            for s in range(seq_len):
                row = []
                for h in range(num_heads):
                    idx = b * num_heads + h
                    row.extend(x.data[idx][s])
                rows.append(row)
            combined.append(rows)
        return Tensor(combined, requires_grad=x.requires_grad)

    def transpose_for_scores(self, x):
        transposed = []
        for mat in x.data:
            transposed.append([list(col) for col in zip(*mat)])
        return Tensor(transposed, requires_grad=x.requires_grad)

    def __call__(self, x):
        print(">>> ENTERING ATTENTION <<<")
        print("x.shape =", x.shape())

        if len(x.shape()) != 3:
            raise ValueError("Expected shape (batch, seq_len, embed_dim)")

        batch_size = x.shape()[0]

        Q = x.matmul(self.W_q)
        K = x.matmul(self.W_k)
        V = x.matmul(self.W_v)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        K_T = self.transpose_for_scores(K)

        scores = Q.matmul(K_T) * (1.0 / math.sqrt(self.head_dim))
        attn_weights = softmax(scores)
        out = attn_weights.matmul(V)

        out = self.combine_heads(out, batch_size)
        return out.matmul(self.W_o)


class FeedForward:
    def __init__(self, embed_dim, hidden_dim):
        def init_tensor(m, n):
            return Tensor([
                [(random.random() * 0.02 - 0.01) for _ in range(n)]
                for _ in range(m)
            ], requires_grad=True)

        self.W1 = init_tensor(embed_dim, hidden_dim)
        self.b1 = Tensor([[0.0] * hidden_dim], requires_grad=True)
        self.W2 = init_tensor(hidden_dim, embed_dim)
        self.b2 = Tensor([[0.0] * embed_dim], requires_grad=True)

    def __call__(self, x):
        shape = x.shape()
        if len(shape) != 3:
            raise ValueError("FeedForward expects input of shape (batch, seq_len, embed_dim)")

        batch, seq_len, dim = shape

        # Step 1: Flatten input for matmul
        flat_input = []
        for b in range(batch):
            for s in range(seq_len):
                flat_input.append(x.data[b][s])

        flat_tensor = Tensor(flat_input, requires_grad=x.requires_grad)

        # Step 2: Fully connected layer 1 + GELU
        dot1 = flat_tensor.matmul(self.W1)
        bias1 = [self.b1.data[0] for _ in range(len(dot1.data))]
        out1 = Tensor([
            [dot1.data[i][j] + bias1[i][j] for j in range(len(dot1.data[0]))]
            for i in range(len(dot1.data))
        ], requires_grad=True)

        activated = gelu(out1)

        # Step 3: Fully connected layer 2 + bias
        dot2 = activated.matmul(self.W2)
        bias2 = [self.b2.data[0] for _ in range(len(dot2.data))]
        out2 = Tensor([
            [dot2.data[i][j] + bias2[i][j] for j in range(len(dot2.data[0]))]
            for i in range(len(dot2.data))
        ], requires_grad=True)

        # Step 4: Reshape back to (batch, seq_len, embed_dim)
        reshaped = []
        flat_idx = 0
        for b in range(batch):
            batch_seq = []
            for s in range(seq_len):
                batch_seq.append(out2.data[flat_idx])
                flat_idx += 1
            reshaped.append(batch_seq)

        return Tensor(reshaped, requires_grad=out2.requires_grad)



class TransformerEncoderBlock:
    def __init__(self, embed_dim, num_heads, hidden_dim):
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, hidden_dim)
        self.norm2 = LayerNorm(embed_dim)

    def __call__(self, x):
        print(">>> ENTER BLOCK: x.shape =", x.shape())
        attn = self.attention(x)
        print(">>> After Attention: shape =", attn.shape())

        x = self.norm1(x + attn)
        print(">>> After Norm1: shape =", x.shape())

        ffn = self.ffn(x)
        print(">>> After FFN: shape =", ffn.shape())

        x = self.norm2(x + ffn)
        print(">>> After Norm2: shape =", x.shape())
        return x

