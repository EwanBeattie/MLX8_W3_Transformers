import data
from models import Transformer
from trainer import Trainer
from config import config
import torch


class Controller():
    def __init__(self):
        train_loader, test_loader = data.get_mnist_data(batch_size=config['batch_size'])
        self.model = Transformer(num_patches=config['num_patches'], patch_dim=7, embedding_size=config['embedding_size'], num_layers=config['num_layers'])
        self.train_loader = train_loader
        self.test_loader = test_loader

        torch.manual_seed(42)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def train_model(self):
        # Initialise the trainer
        trainer = Trainer(self.model, self.train_loader, self.test_loader).to(self.device)

        # Run the trainer
        trainer.train()


    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
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
