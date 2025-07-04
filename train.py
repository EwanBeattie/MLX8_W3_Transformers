import torch
from math import sqrt
import data
from models import Transformer
from configs import hyperparameters, run_config, sweep_config
import wandb
import torch.nn as nn
from tqdm import tqdm

def main():

    torch.manual_seed(42)
    
    if run_config['run_type'] == 'sweep':
        sweep_id = wandb.sweep(sweep_config, entity=run_config['entity'], project=run_config['project'])
        wandb.agent(
            sweep_id=sweep_id,
            function=train,
            project=run_config['project'],
            count=40,
        )
    elif run_config['run_type'] == 'train':
        trained_weights = train(hyperparameters)
        torch.save(trained_weights, "model_weights.pth")


def train(config=None):
    # Obtain correct cofig dict, init wandb then create wandb.config object
    if config is None:
        config = sweep_config
    wandb.init(entity=run_config['entity'], project=run_config['project'], config=config)
    config = wandb.config

    device = get_device()
    
    train_loader, test_loader = data.get_mnist_data(batch_size=config.batch_size)

    # Check the image can be diveded into patches of predetermined size
    num_patches = config.num_patches
    image_dim = 28
    if int(sqrt(num_patches)) ** 2 != num_patches:
        raise ValueError("num_patches must be a perfect square.")
    if image_dim % int(sqrt(num_patches)) != 0:
        raise ValueError("Number of patches must be a perfect square that divides the image dimension evenly.")
    patch_dim = int(28 / sqrt(num_patches))

    # Initialise the model
    model = Transformer(
        num_patches=num_patches,
        patch_dim=patch_dim,
        embedding_size=config.embedding_size,
        num_layers=config.num_layers
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(config.epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch_index, (data_batch, target) in loop:
            data_batch = data_batch.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data_batch)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()

            if batch_index % 100 == 0:
                print(f'Epoch: {epoch + 1}, Batch: {batch_index}, Loss: {loss.item():.2f}')
                wandb.log({'loss': loss.item()})
            loop.set_postfix(loss=loss.item())

    test(model, test_loader, device)

    wandb.finish()

    return model.state_dict()

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data_batch, target in tqdm(test_loader, desc="Testing", total=len(test_loader)):
            data_batch = data_batch.to(device)
            target = target.to(device)

            output = model(data_batch)
            _, predicted = output.max(1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    wandb.log({"test_accuracy": accuracy})


if __name__ == "__main__":
    main()
