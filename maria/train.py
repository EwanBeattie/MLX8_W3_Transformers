import torch, wandb, datetime, tqdm
from torch import nn
from main import MNISTDataProcessor, TransformerEncoderBlock, PatchEncoderClassifier

# Set up environment and tracking
wandb.init(project='transformer-mnist')
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
torch.os.makedirs('./checkpoints', exist_ok=True)

# Embedding and model setup
embed = nn.Linear(49, 128).to(device)  # for 7x7 patches
encoder = nn.Sequential(*[TransformerEncoderBlock(embed_dim=128, num_heads=1) for _ in range(6)]).to(device)
model = PatchEncoderClassifier(encoder, embed_dim=128, num_classes=10).to(device)

# Optimizer and loss
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
torch.save(model.state_dict(), f'./checkpoints/{ts}.0.pth')
print('model params:', sum(p.numel() for p in model.parameters()))

# Load dataset
processor = MNISTDataProcessor()
train_loader = processor.train_loader

# Training loop
for epoch in range(5):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        patches = processor.extract_patches_batch(images).to(device)  # [B, 16, 49]
        patch_tokens = embed(patches)                                 # [B, 16, 128]

        outputs = model(patch_tokens)
        loss = loss_fn(outputs, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        wandb.log({'loss': loss.item()})

    acc = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={acc:.2f}%")
    torch.save(model.state_dict(), f'./checkpoints/{ts}.{epoch+1}.pth')

wandb.finish()