```python
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from src.model import GravitationalWaveModel
from src.data_loader import GravitationalWaveDataset
from src.utils import calculate_accuracy
from src.config import Config

def train_model():
    # Load the configuration
    config = Config()

    # Load the data
    dataset = GravitationalWaveDataset(config.data_path)
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize the model
    model = GravitationalWaveModel()
    model.to(config.device)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    for epoch in range(config.epochs):
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(config.device), labels.to(config.device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            accuracy = calculate_accuracy(outputs, labels)

            # Print training progress
            if (i+1) % config.print_every == 0:
                print(f'Epoch [{epoch+1}/{config.epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item()}, Accuracy: {accuracy}')

    # Save the trained model
    torch.save(model.state_dict(), config.model_path)

if __name__ == "__main__":
    train_model()
```