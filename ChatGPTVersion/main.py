import torch
from data import get_mnist_data
from models import SimpleTransformerEncoder
from trainer import Trainer
from config import config

def main():
    train_loader, test_loader = get_mnist_data(batch_size=config['batch_size'])
    model = SimpleTransformerEncoder(
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    trainer = Trainer(model, train_loader, test_loader, config)
    trainer.train()
    trainer.evaluate()

if __name__ == '__main__':
    main()