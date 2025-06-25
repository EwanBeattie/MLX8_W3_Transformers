from torchvision import datasets
from torchvision.transforms import ToTensor

train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=ToTensor())

print(f"Training set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")