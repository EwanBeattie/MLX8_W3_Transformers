import torch
import torch.optim as optim
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, train_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    def train_step(self, data, target):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        for epoch in range(self.config['epochs']):
            epoch_loss = 0
            for data, target in self.train_loader:
                loss = self.train_step(data, target)
                epoch_loss += loss
            print(f"Epoch [{epoch+1}/{self.config['epochs']}], Loss: {epoch_loss/len(self.train_loader)}")

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.cuda(), target.cuda()
                output = self.model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
