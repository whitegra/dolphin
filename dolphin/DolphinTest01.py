import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import brown

from dolphin.tensor import Tensor
from dolphin.activations import softmax, cross_entropy_loss
from dolphin.transformers import TransformerEncoderBlock
from dolphin.optimizers import Adam


def main():
    # 1. load & tokenize dataset
    
    nltk.download("brown")
    nltk.download("punkt")

    sentences = brown.sents()[:500]
    sentences = [" ".join(sent) for sent in sentences]
    tokenized = [word_tokenize(s.lower()) for s in sentences]


    # 2. build vocab
    
    all_words = [w for s in tokenized for w in s]
    freq = {}
    for word in all_words:
        freq[word] = freq.get(word, 0) + 1
    vocab = sorted(freq, key=freq.get, reverse=True)[:1000]
    word_to_index = {w: i for i, w in enumerate(vocab)}
    index_to_word = {i: w for w, i in word_to_index.items()}
    vocab_size = len(word_to_index)

    # 3. prepare training data
    seq_len = 4
    x_data, y_data = [], []

    for s in tokenized:
        idxs = [word_to_index[w] for w in s if w in word_to_index]
        for i in range(len(idxs) - seq_len):
            x_data.append(idxs[i:i + seq_len])
            y_data.append([idxs[i + seq_len]])

    x_train = Tensor(x_data)  # shape: (batch, seq_len)
    y_train = Tensor(y_data)  # shape: (batch, 1)


    # 4. config model
    embed_dim = 32
    num_heads = 2
    hidden_dim = 64
    num_layers = 2
    epochs = 2
    lr = 0.01

    # 5. embedings / embed input
    embedding_weights = Tensor([
        [random.uniform(-0.01, 0.01) for _ in range(embed_dim)]
        for _ in range(vocab_size)
    ], requires_grad=True)

    def embed_input(x):
        if len(x.shape()) != 2:
            raise ValueError(f"Expected shape (batch, seq_len), got {x.shape()}")
        batch, seq_len = x.shape()
        embedded = []
        for row in x.data:
            embedded_row = [embedding_weights.data[int(idx)] for idx in row]
            embedded.append(embedded_row)
        return Tensor(embedded, requires_grad=True)

    # 6. transformer model and encoders: 
        
    encoder = [TransformerEncoderBlock(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)]

    def transformer_forward(x):
        for block in encoder:
            x = block(x)
        # Project to vocab
        weight_T = [[embedding_weights.data[r][c] for r in range(vocab_size)] for c in range(embed_dim)]
        projected = []
        for batch in x.data:
            proj_batch = []
            for vec in batch:
                proj_vec = [sum(vec[i] * weight_T[i][j] for i in range(embed_dim)) for j in range(vocab_size)]
                proj_batch.append(proj_vec)
            projected.append(proj_batch)
        return Tensor(projected, requires_grad=True)


    # 7. optimizer
    params = [embedding_weights] + [v for block in encoder for v in vars(block).values() if isinstance(v, Tensor)]
    optimizer = Adam(params, lr=lr)

    # 8. train!
    print("\nTraining Dolphin Transformer...\n")
    for epoch in range(epochs):
        x_embed = embed_input(x_train)
        logits = transformer_forward(x_embed)
        final_logits = Tensor([logit[-1] for logit in logits.data])  # only last token
        loss = cross_entropy_loss(final_logits, y_train)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.data[0][0]:.4f}")

    print("\nTraining complete.\n")

    # 9. GENERATE TEXT! 
    
    def generate_text(start_word, length=5):
        sequence = [word_to_index.get(start_word.lower(), 0)]
        for _ in range(length):
            context = sequence[-seq_len:]
            padded = [0] * (seq_len - len(context)) + context
            input_tensor = Tensor([padded])
            x_embed = embed_input(input_tensor)
            logits = transformer_forward(x_embed)
            last_logits = logits.data[0][-1]
            probs = softmax(Tensor([last_logits])).data[0]
            next_idx = random.choices(range(vocab_size), weights=probs, k=1)[0]
            sequence.append(next_idx)
        return " ".join(index_to_word.get(i, "<unk>") for i in sequence)


    # 10. results !!!!
    print("\nGenerated Text:")
    print(generate_text("the"))
    print(generate_text("science"))
    print(generate_text("they"))


# this ensures the script only runs once when executed directly, will take somee time. 
if __name__ == "__main__":
    main()
