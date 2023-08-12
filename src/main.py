```python
import torch
from src.data_loader import DataLoader
from src.model import Model
from src.train import train_model
from src.utils import *
from src.config import Config

def main():
    # Load configurations
    config = Config()

    # Initialize data loader
    data_loader = DataLoader(config.data_path)

    # Load data
    train_data, test_data = data_loader.load_data()

    # Initialize model
    model = Model(config)

    # Check if CUDA is available and if so, use it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Train the model
    train_model(model, train_data, config, device)

    # Evaluate the model
    test_loss, test_accuracy = evaluate_model(model, test_data, device)

    # Print test loss and accuracy
    print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_accuracy:.3f}')

    # Save the trained model
    torch.save(model.state_dict(), config.model_save_path)

if __name__ == "__main__":
    main()
```