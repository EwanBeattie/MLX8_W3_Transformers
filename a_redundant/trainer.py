from configs import hyperparameters, sweep_config, run_config
import torch
import torch.nn as nn
import wandb


class Trainer:
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hyperparameters['learning_rate'])
        self.loss_function = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, wandb_config=None):

        if wandb_config == None:
            wandb_config = hyperparameters

        wandb.init(entity=run_config['entity'], project=run_config['project'], config=wandb_config)

        self.model.train()
        for epoch in range(hyperparameters['epochs']):
            for batch_index, (data, target) in enumerate(self.train_loader):
                data =  data.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()

                output = self.model(data)

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

    def run_sweep():
        sweep_id = wandb.sweep(sweep_config, entity=run_config['entity'], project=run_config['project'])

        wandb.agent(
            sweep_id=sweep_id,
            function=train,
            project=run_config['project'],
            count=40,
        )