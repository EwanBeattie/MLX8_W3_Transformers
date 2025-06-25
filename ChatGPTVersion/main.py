import torch
from data import get_mnist_data
from models import SimpleTransformerEncoder
from a_redundant.trainer import Trainer
from configs import hyperparameters

def main():
    train_loader, test_loader = get_mnist_data(batch_size=hyperparameters['batch_size'])
    model = SimpleTransformerEncoder(
        embed_dim=hyperparameters['embed_dim'],
        num_heads=hyperparameters['num_heads'],
        num_layers=hyperparameters['num_layers'],
        dropout=hyperparameters['dropout']
    )
    trainer = Trainer(model, train_loader, test_loader, hyperparameters)
    trainer.train()
    trainer.evaluate()

if __name__ == '__main__':
    main()