import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import sentencepiece as spm


class MNISTDataProcessor:
    # 1. Define the transform to convert images to tensors and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),  
        # 📸 Converts image from a PIL image or NumPy array into a PyTorch tensor
        # 🧭 Reorders shape from (Height, Width, Channels) → (Channels, Height, Width)
        # 🎚️ Scales pixel values from [0, 255] to [0.0, 1.0] (important for training)
    ])

    def __init__(self, batch_size_train=64, batch_size_test=1000, patch_size=7):
        # 2. Load the training and test datasets

        # 🔹 Load the training data:
        self.train_dataset = datasets.MNIST(
            root="./data",             # Folder where the data will be stored/downloaded
            train=True,                # True → this is the training set (60,000 images)
            download=True,             # Download the data if it doesn't already exist
            transform=self.transform   # Apply the transform (tensor + normalization)
        )

        # 🔹 Load the test data:
        self.test_dataset = datasets.MNIST(
            root="./data",             # Same folder for consistency
            train=False,               # False → this is the test set (10,000 images)
            download=True,             # Also download if not already present
            transform=self.transform   # Apply the same transform for consistency
        )

        # 3. Create DataLoaders to load the data in batches

        # 🔄 Loads training data in batches of 64 samples
        # ✅ shuffle=True ensures data is shuffled each epoch → improves generalization and avoids learning order bias
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size_train, shuffle=True)

        # 🔍 Loads test data in one large batch of 1000
        # 🚫 shuffle=False (default) → keeps test set in fixed order for consistent evaluation
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size_test)

        self.patch_size = patch_size

    # 🔧 Function to extract patches from each image in a batch
    def extract_patches_batch(self, images, patch_size=None):
        if patch_size is None:
            patch_size = self.patch_size
        """
        Splits a batch of images [batch_size, 1, 28, 28] into non-overlapping patches.
        Returns a tensor of shape [batch_size, num_patches, patch_size*patch_size].
        """

        # Get how many images are in the batch
        batch_size = images.size(0)

        # 🧩 Unfold the image to create 2D patches:
        # Split each 28x28 image into 16 non-overlapping patches of size 7x7.
        # This creates 4 patches across the height and 4 across the width.
        
        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

        # Flatten each 7x7 patch into a 1D vector (49 values)
        # .contiguous() ensures memory is laid out correctly for .view()
        # Result: [batch_size, num_patches, 49]
        patches = patches.contiguous().view(batch_size, -1, patch_size * patch_size)
        # Final shape: [batch_size, 16, 49] → 16 patches, each represented as a 49-dim vector

        return patches  # 🔁 Returns the sequence of flattened patches per image



class SingleHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # Linear layers to compute Query, Key, Value
        self.query_linear = nn.Linear(embed_dim, embed_dim)  # Transforms input into Query vectors
        self.key_linear = nn.Linear(embed_dim, embed_dim)    # Transforms input into Key vectors
        self.value_linear = nn.Linear(embed_dim, embed_dim)  # Transforms input into Value vectors

        # Output projection layer to mix the output from attention mechanism (optional but common)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout layer for regularization (helps prevent overfitting)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]

        # Compute Q, K, V matrices - Each produces a new tensor with the same shape as x
        Q = self.query_linear(x)  # Query: Represents what you're searching for in the input. [batch_size, seq_len, embed_dim]
        K = self.key_linear(x)    # Key: Represents the features of each token [batch_size, seq_len, embed_dim]
        V = self.value_linear(x)  # Value: Contains the actual data that will be returned to be passed. [batch_size, seq_len, embed_dim]

        # Compute attention scores: Q x K^T
        scores = torch.matmul(Q, K.transpose(-2, -1))  # Multiply Q by the flipped K to get scores showing how much each part should pay attention to others.

        # Scale scores by sqrt(embed_dim)
        scores = scores / (self.embed_dim ** 0.5)  # Prevents overly large dot products

        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # Converts raw attention scores into normalized weights (probabilities). These weights decide how much attention each token pays to others.
        attn_weights = self.dropout(attn_weights) # Dropout for regularization

        # Weighted sum of values
        context = torch.matmul(attn_weights, V)   # Use attention weights to mix the value vectors, gathering important information from all tokens.

        # Output projection to bring context back into embedding space
        output = self.out_proj(context)

        return output, attn_weights



class FeedForward(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion_factor),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * expansion_factor, embed_dim)
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = SingleHeadSelfAttention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x



class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=7, embed_dim=128, img_channels=1, img_size=28):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2  # total patches per image

        # Each patch has size patch_size * patch_size * channels (flattened)
        self.patch_dim = patch_size * patch_size * img_channels

        # Linear layer to embed each flattened patch into embed_dim vector
        self.linear_proj = nn.Linear(self.patch_dim, embed_dim)

        # Learnable positional embeddings for each patch position
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

    def forward(self, x):
        """
        x: tensor of shape [batch_size, channels, height, width]
        """
        B = x.shape[0]

        # Extract patches using unfold
        patches = x.unfold(2, self.patch_size, self.patch_size) \
                   .unfold(3, self.patch_size, self.patch_size)
        # [B, C, H//P, W//P, P, P]

        # Rearrange and flatten each patch
        patches = patches.permute(0, 2, 3, 1, 4, 5)  # [B, num_patches_h, num_patches_w, C, P, P]
        patches = patches.flatten(1, 2)             # [B, num_patches, C, P, P]
        patches = patches.flatten(2)                # [B, num_patches, patch_dim]

        # Project to embedding dimension
        embeddings = self.linear_proj(patches)      # [B, num_patches, embed_dim]

        # Add positional encoding
        embeddings = embeddings + self.pos_embedding

        return embeddings



class TransformerEncoder(nn.Module):
    def __init__(self, patch_embedder, embed_dim=128, num_heads=1, num_layers=6, num_classes=10):
        super().__init__()
        self.patch_embedder = patch_embedder
        self.pos_embedding = nn.Parameter(torch.randn(1, patch_embedder.num_patches, embed_dim))  # [1, num_patches, embed_dim]

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embedder(x)  # [B, num_patches, embed_dim]

        # Add position embedding
        x = x + self.pos_embedding

        # Pass through encoder blocks
        for layer in self.encoder_layers:
            x = layer(x)

        # Final normalization
        x = self.final_norm(x)

        # Mean pooling over patch tokens
        pooled = x.mean(dim=1)  # [B, embed_dim]

        return self.classifier(pooled)  # [B, num_classes]
    
embed = nn.Linear(196, 128)  # if 14x14 patches

class PatchEncoderClassifier(nn.Module):
    def __init__(self, encoder, embed_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        x = self.encoder(x)                # [batch, num_patches, embed_dim]
        pooled = x.mean(dim=1)             # [batch, embed_dim]
        logits = self.classifier(pooled)   # [batch, num_classes]
        return logits