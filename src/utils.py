```python
import os
import time
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))

def calculate_time(start_time):
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return cm, report

def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std
```