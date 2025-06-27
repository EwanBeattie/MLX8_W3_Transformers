# import wandb
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from main import PatchEmbedder, Encoder

# # --- Model definition (must match training exactly) ---
# class PatchEncoderClassifier(nn.Module):
#     def __init__(self, patch_embedder, encoder, embed_dim, num_classes):
#         super().__init__()
#         self.patch_embedder = patch_embedder
#         self.encoder = encoder
#         self.classifier = nn.Linear(embed_dim, num_classes)

#     def forward(self, x):
#         patches = self.patch_embedder(x)
#         encoded = self.encoder(patches)
#         pooled = encoded.mean(dim=1)
#         logits = self.classifier(pooled)
#         return logits


# wandb.define_metric("eval_step")  # register it as a valid step
# wandb.define_metric("eval_batch_loss", step_metric="eval_step")  # use as x-axis

# # --- Initialize wandb ---
# wandb.init(
#     project="transformer-eval", 
#     config={
#     "embed_dim": 64,
#     "num_heads": 4,
#     "num_layers": 2,
#     "batch_size": 64,
#     "num_classes": 10
# })

# wandb.define_metric("eval_batch_loss", step_metric="eval_step")
# wandb.define_metric("final_test_loss")
# wandb.define_metric("final_test_accuracy")

# # --- Device configuration ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # --- Reconstruct exact model ---
# patch_embedder = PatchEmbedder(embed_dim=wandb.config.embed_dim)
# encoder = Encoder(
#     embed_dim=wandb.config.embed_dim,
#     num_heads=wandb.config.num_heads,
#     num_layers=wandb.config.num_layers
# )
# model = PatchEncoderClassifier(patch_embedder, encoder, wandb.config.embed_dim, wandb.config.num_classes).to(device)

# # --- Load saved trained weights ---
# model_path = f'patch_encoder_classifier_{wandb.config.embed_dim}_{wandb.config.num_layers}_{wandb.config.num_heads}.pth'
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.eval()

# # --- Load MNIST test data ---
# transform = transforms.Compose([
#     transforms.ToTensor()
# ])
# test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
# test_loader = DataLoader(test_dataset, batch_size=wandb.config.batch_size)

# # --- Evaluation loop ---
# correct, total, total_loss = 0, 0, 0.0

# with torch.no_grad():
#     for step, (images, labels) in enumerate(test_loader):
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         loss = F.cross_entropy(outputs, labels)
#         total_loss += loss.item() * labels.size(0)

#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#         # 🪵 Log batch-level loss to wandb
#         wandb.log({"eval_batch_loss": loss.item(), "eval_step": step})

# # --- Metrics calculation ---
# avg_loss = total_loss / total
# accuracy = correct / total

# # --- Log metrics to wandb ---
# # wandb.log({
# #     "epoch": 1,
# #     "test_accuracy": accuracy,
# #     "test_loss": avg_loss
# # })

# wandb.log({
#     "final_test_accuracy": accuracy,
#     "final_test_loss": avg_loss
# })

# # --- Print evaluation results ---
# print(f"✅ Test Accuracy: {accuracy:.4f}")
# print(f"📉 Test Loss: {avg_loss:.4f}")

# wandb.finish()

# # steps 


# # wand.init 


# # wandb.log({"eval_step": step})


# # wandb.log({"eval_batch_loss": loss.item(), "eval_step": step})




import torch

torch.manual_seed(42)
x = torch.randn(3)
print(x)