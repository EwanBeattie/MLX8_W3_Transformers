import torch
import torch.nn as nn
from layers import MultiHeadSelfAttention, FeedForwardLayer

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, image_size=28, patch_size=7, num_classes=10, embed_dim=256, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size
        self.patch_embedding = nn.Linear(self.patch_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.encoder_layers = nn.ModuleList([MultiHeadSelfAttention(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.feedforward_layers = nn.ModuleList([FeedForwardLayer(embed_dim, dropout) for _ in range(num_layers)])
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, self.num_patches, self.patch_dim)
        x = self.patch_embedding(patches)
        x += self.positional_encoding
        for attention, feedforward in zip(self.encoder_layers, self.feedforward_layers):
            x = attention(x)
            x = feedforward(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x
