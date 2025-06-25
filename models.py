import torch.nn as nn
import torch
from config import config

class Transformer(nn.Module):
    def __init__(self, num_patches, patch_dim, embedding_size, num_layers):
        
        super().__init__()
        self.num_layers = num_layers
        self.num_patches = num_patches
        self.patch_dim = patch_dim

        self.embedding = nn.Linear(patch_dim**2, embedding_size)
        self.positional_encoding = nn.Embedding(num_patches, embedding_size)
        self.encoding_layers = nn.ModuleList([Encoder(embedding_size) for _ in range(num_layers)])
        self.classifier = nn.Linear(embedding_size, 10)


    def forward(self, x):
        # Patch management
        batch_size = x.size(0)
        patches = x.unfold(2, self.patch_dim, self.patch_dim).unfold(3, self.patch_dim, self.patch_dim)
        patches = patches.contiguous().view(batch_size, self.num_patches, self.patch_dim **2)

        # Turn from patch to embedding matrix
        x = self.embedding(patches)

        x += self.positional_encoding()

        # Pass through the encoder
        for i in range(self.num_layers):
            x = self.encoding_layers[i](x)

        # Avg pooling
        x = x.mean(dim=1)

        # Pass through the classifier
        x = self.classifier(x)
        
        return x
    

class Encoder(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.attention = AttentionLayer(self.embedding_size, key_query_size=config['key_query_size'], value_size=config['value_size'])
        self.mlp = MLPLayer(self.embedding_size)

    def forward(self, x):
        x = self.attention(x)
        x = self.mlp(x)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, embedding_size, key_query_size, value_size):
        super().__init__()
        self.query = nn.Linear(embedding_size, key_query_size)
        self.key = nn.Linear(embedding_size, key_query_size)
        self.value = nn.Linear(embedding_size, value_size)
        self.softmax = nn.Softmax(-1)
        self.weights = nn.Linear(value_size, embedding_size)
        self.drop = torch.nn.Dropout(0.1)


    def forward(self, embedding_matrix):
        # Embedding matrix is of shape (batch_size, num_patches, embedding_size)

        # Multiply E (embedding matrix) with W_query to get Q
        Q = self.query(embedding_matrix)
        # Multiply E (embedding matrix) with W_key to get K
        K = self.key(embedding_matrix)
        # Multiply Q with K transposed, normalise with d_k and softmax to get A
        A = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        # Softmax the attention scores
        A = self.softmax(A)
        # Apply dropout to the attention scores
        A = self.drop(A)

        # Multiply E (embedding matrix) with W_value to get V
        V = self.value(embedding_matrix)
        # Multiply A with V to get the hidden states
        H = torch.matmul(A, V)
        # Multiply H with W_weights to get the final output
        x = self.weights(H)

        return x

class MLPLayer(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.fc1 = nn.Linear(embedding_size, 128)
        self.drop = torch.nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, embedding_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x