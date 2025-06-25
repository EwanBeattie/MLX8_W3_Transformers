import data
from models import Transformer
from trainer import Trainer
from config import config
import torch
from math import sqrt


class Controller():
    def __init__(self):
        torch.manual_seed(42)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_loader, test_loader = data.get_mnist_data(batch_size=config['batch_size'])
        patch_dim = int(28 / sqrt(config['num_patches']))
        self.model = Transformer(num_patches=config['num_patches'], 
                                 patch_dim=patch_dim, 
                                 embedding_size=config['embedding_size'], 
                                 num_layers=config['num_layers'])
        
        self.model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train_model(self):
        # Initialise the trainer
        trainer = Trainer(self.model, self.train_loader, self.test_loader)

        # Run the trainer
        trainer.train()


    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data =  data.to(self.device)
                target = target.to(self.device)

                output = self.model(data)
                _, predicted = output.max(1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    controller = Controller()

    # Train and test
    controller.train_model()
    controller.test()

    # Load weights and test
    # controller.model.load_state_dict(torch.load("model_weights.pth"))
    # controller.test()
