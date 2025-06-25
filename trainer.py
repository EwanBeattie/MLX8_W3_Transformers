from config import config
import torch
import torch.nn as nn
import wandb


class Trainer:
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.loss_function = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    def train(self):
        wandb.init(entity='ewanbeattie1-n-a', project='mlx8-week-03-transformers', config=config)

        self.model.train()
        for epoch in range(config['epochs']):
            for batch_index, (data, target) in enumerate(self.train_loader):
                data =  data.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()

                output = self.model(data).to(self.device)

                # Compute the loss
                loss = self.loss_function(output, target)

                # Back pass
                loss.backward()
                # What is optimisation?
                self.optimizer.step()


                if batch_index % 100 == 0:
                    print(f'Epoch: {epoch + 1}, Batch: {batch_index}, Loss: {loss.item():.2f}')

                wandb.log({'loss': loss.item()})


        
        torch.save(self.model.state_dict(), "model_weights.pth")
        wandb.finish()