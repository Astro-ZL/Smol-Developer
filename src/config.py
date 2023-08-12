```python
# src/config.py

# Importing necessary libraries
import os

# Defining the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Defining the path for the PTA gravitational wave data
DATA_PATH = os.path.join(BASE_DIR, 'data/gravitational_wave_data.csv')

# Defining the model save path
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models/')

# Defining the hyperparameters for the model
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 100

# Defining the train-test split ratio
TRAIN_TEST_SPLIT = 0.8

# Defining the seed for reproducibility
SEED = 42
```