import torch
import torch.nn as nn
import math
from torch.nn.functional import log_softmax
import torch.optim as optim

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Adjust mask shape to match attn_scores
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            # Ensure the mask is broadcastable
            mask = mask.expand(-1, self.num_heads, -1, -1)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.W_o(attn_output)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        print("x shape:", x.shape)
        print("mask shape:", mask.shape)
        
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = self.create_positional_encoding(max_seq_length, d_model)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def create_positional_encoding(self, max_seq_length, d_model):
        pos_encoding = torch.zeros(max_seq_length, d_model)
        positions = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pos_encoding[:, 0::2] = torch.sin(positions * div_term)
        pos_encoding[:, 1::2] = torch.cos(positions * div_term)
        return pos_encoding.unsqueeze(0)
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        src_embedded = self.dropout(self.encoder_embedding(src) + self.positional_encoding[:, :src.size(1), :])
        tgt_embedded = self.dropout(self.decoder_embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :])
        
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        
        output = self.fc(dec_output)
        return output

# Example usage
src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# Generate random input data
src = torch.randint(1, src_vocab_size, (64, 20))  # (batch_size, src_seq_len)
tgt = torch.randint(1, tgt_vocab_size, (64, 15))  # (batch_size, tgt_seq_len)
src_mask = torch.ones(64, 1, 20)  # (batch_size, 1, src_seq_len)
tgt_mask = torch.tril(torch.ones(64, 15, 15))  # (batch_size, tgt_seq_len, tgt_seq_len)

output = transformer(src, tgt, src_mask, tgt_mask)
print(output.shape)  # Should be (64, 15, tgt_vocab_size)

# Define a small vocabulary for English and French
eng_vocab = ["<pad>", "<sos>", "<eos>", "hello", "world", "how", "are", "you"]
fra_vocab = ["<pad>", "<sos>", "<eos>", "bonjour", "monde", "comment", "allez", "vous"]

# Create dictionaries for word to index mapping
eng_word2idx = {word: idx for idx, word in enumerate(eng_vocab)}
fra_word2idx = {word: idx for idx, word in enumerate(fra_vocab)}

# Example sentences
eng_sentences = [
    "hello world",
    "how are you",
    "hello how are you"
]
fra_sentences = [
    "bonjour monde",
    "comment allez vous",
    "bonjour comment allez vous"
]

# Function to convert sentence to tensor
def sentence_to_tensor(sentence, word2idx, max_length):
    words = sentence.split()
    tensor = torch.zeros(max_length, dtype=torch.long)
    tensor[0] = word2idx['<sos>']
    for i, word in enumerate(words, start=1):
        tensor[i] = word2idx[word]
    tensor[len(words)+1] = word2idx['<eos>']
    return tensor

# Prepare data
max_length = 10  # Maximum sentence length including <sos> and <eos>
src_tensors = [sentence_to_tensor(sent, eng_word2idx, max_length) for sent in eng_sentences]
tgt_tensors = [sentence_to_tensor(sent, fra_word2idx, max_length) for sent in fra_sentences]

# Model parameters
src_vocab_size = len(eng_vocab)
tgt_vocab_size = len(fra_vocab)
d_model = 32  # Reduced for this small example
num_heads = 2
num_layers = 2
d_ff = 64
max_seq_length = max_length
dropout = 0.1

# Create the model
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# Training loop
criterion = nn.CrossEntropyLoss(ignore_index=fra_word2idx['<pad>'])
optimizer = optim.Adam(transformer.parameters(), lr=0.0001)

num_epochs = 100
for epoch in range(num_epochs):
    transformer.train()
    total_loss = 0
    for src, tgt in zip(src_tensors, tgt_tensors):
        src = src.unsqueeze(0)  # Add batch dimension
        tgt = tgt.unsqueeze(0)  # Add batch dimension
        
        src_mask = (src != eng_word2idx['<pad>']).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != fra_word2idx['<pad>']).unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_mask & torch.tril(torch.ones((tgt.size(-1), tgt.size(-1)), device=tgt.device)).bool()

        optimizer.zero_grad()
        output = transformer(src, tgt[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1])
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(src_tensors)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Testing
transformer.eval()
fra_idx2word = {idx: word for word, idx in fra_word2idx.items()}

def translate(sentence):
    transformer.eval()
    src = sentence_to_tensor(sentence, eng_word2idx, max_length).unsqueeze(0)
    src_mask = (src != eng_word2idx['<pad>']).unsqueeze(1).unsqueeze(2)
    
    with torch.no_grad():
        output = transformer(src, src, src_mask, None)
        output = torch.argmax(output, dim=-1)
    
    return ' '.join([fra_idx2word[idx.item()] for idx in output[0] if idx.item() not in [fra_word2idx['<pad>'], fra_word2idx['<sos>'], fra_word2idx['<eos>']]])

# Test on training data
print("\nTest Results:")
for eng, fra in zip(eng_sentences, fra_sentences):
    predicted = translate(eng)
    print(f"English: {eng}")
    print(f"French (actual): {fra}")
    print(f"French (predicted): {predicted}")
    print()

# Test on new data
new_sentences = [
    "hello you",
    "how are",
    "world how"
]

print("\nTest on New Data:")
for eng in new_sentences:
    predicted = translate(eng)
    print(f"English: {eng}")
    print(f"French (predicted): {predicted}")
    print()