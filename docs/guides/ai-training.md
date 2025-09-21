# AI Model Training Guide

This guide covers how to train, evaluate, and deploy machine learning models in the AIPlatform ecosystem, with a focus on privacy-preserving techniques like federated learning.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Setting Up the Environment](#setting-up-the-environment)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Federated Learning](#federated-learning)
- [Model Evaluation](#model-evaluation)
- [Model Packaging](#model-packaging)
- [Deployment](#deployment)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.8+
- PyTorch 2.0+ or TensorFlow 2.12+
- CUDA (for GPU acceleration)
- Docker (optional)
- Basic understanding of machine learning concepts

## Project Structure

```
ai/
├── data/               # Data loading and preprocessing
│   ├── loaders.py
│   └── preprocessing.py
├── models/             # Model architectures
│   ├── __init__.py
│   ├── cnn.py
│   └── transformer.py
├── training/           # Training logic
│   ├── trainer.py
│   └── metrics.py
├── federated/          # Federated learning
│   ├── server.py
│   └── client.py
├── utils/              # Utility functions
│   ├── logger.py
│   └── helpers.py
├── configs/            # Configuration files
│   ├── base.yaml
│   └── federated.yaml
└── scripts/            # Training scripts
    ├── train.py
    └── federated_train.py
```

## Setting Up the Environment

### 1. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install tensorflow
pip install flwr  # Flower for federated learning
pip install numpy pandas scikit-learn matplotlib
pip install pyyaml tqdm
```

### 3. Install AIPlatform SDK

```bash
pip install aiplatform-sdk
```

## Data Preparation

### 1. Data Loading

```python
# data/loaders.py
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

class AIDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.samples = self._load_samples()
    
    def _load_samples(self):
        # Load your data here
        # Return list of (input, target) tuples
        pass
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

def create_dataloaders(data_dir, batch_size=32):
    # Define transforms
    train_transform = Compose([
        RandomHorizontalFlip(),
        RandomRotation(10),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = AIDataset(data_dir, transform=train_transform, split='train')
    val_dataset = AIDataset(data_dir, transform=val_transform, split='val')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader
```

### 2. Data Preprocessing

```python
# data/preprocessing.py
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess an image for model input."""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    return img

def preprocess_tabular(data, scaler=None, fit_scaler=False):
    """Preprocess tabular data with optional scaling."""
    if fit_scaler and scaler is None:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        return data, scaler
    elif scaler is not None:
        return scaler.transform(data), scaler
    return data, None
```

## Model Architecture

### 1. Define a Simple CNN

```python
# models/cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)  # Adjust based on input size
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### 2. Define a Transformer Model

```python
# models/transformer.py
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class TextClassifier(nn.Module):
    def __init__(self, num_classes, model_name='bert-base-uncased'):
        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
```

## Training Process

### 1. Training Loop

```python
# training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}')
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=1.0
            )
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def fit(self, num_epochs):
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, 'best_model.pth')
            
            print(f'Epoch {epoch + 1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n')
        
        return best_val_acc
```

### 2. Training Script

```python
# scripts/train.py
import yaml
import torch
from torchvision import transforms
from data.loaders import create_dataloaders
from models.cnn import SimpleCNN
from training.trainer import Trainer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load config
    config = load_config('configs/base.yaml')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        config['data_dir'],
        batch_size=config['batch_size']
    )
    
    # Initialize model
    model = SimpleCNN(num_classes=config['num_classes'])
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config['training']
    )
    
    # Train model
    best_val_acc = trainer.fit(num_epochs=config['training']['num_epochs'])
    print(f'Best validation accuracy: {best_val_acc:.2f}%')

if __name__ == '__main__':
    main()
```

## Federated Learning

### 1. Server Implementation

```python
# federated/server.py
import flwr as fl
from typing import List, Tuple, Dict, Optional
import torch

class AIServer(fl.server.strategy.FedAvg):
    def __init__(self, model, num_rounds=10, **kwargs):
        self.model = model
        self.num_rounds = num_rounds
        super().__init__(**kwargs)
    
    def configure_fit(self, rnd, parameters, client_manager):
        """Configure the next round of training."""
        config = {
            'current_round': rnd,
            'num_rounds': self.num_rounds,
            'batch_size': 32,
            'epochs': 1,
        }
        return super().configure_fit(rnd, parameters, client_manager, config)
    
    def aggregate_fit(self, rnd, results, failures):
        """Aggregate model updates using weighted average."""
        # Get aggregated weights from FL strategy
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        
        if aggregated_weights is not None:
            # Save aggregated weights to model
            self.model.set_weights(aggregated_weights)
            
            # Save model checkpoint
            torch.save(self.model.state_dict(), f'model_round_{rnd}.pth')
        
        return aggregated_weights

def start_server(model, num_rounds=10, num_clients=2):
    """Start Flower server for federated learning."""
    # Define strategy
    strategy = AIServer(
        model=model,
        num_rounds=num_rounds,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
    )
    
    # Start server
    fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": num_rounds},
        strategy=strategy,
    )
```

### 2. Client Implementation

```python
# federated/client.py
import flwr as fl
import torch
from torch.utils.data import DataLoader

class AIClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        # Set model parameters
        self.set_parameters(parameters)
        
        # Train model
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.get('lr', 0.01),
            momentum=0.9
        )
        
        for epoch in range(config.get('epochs', 1)):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass
                outputs = self.model(x)
                loss = torch.nn.functional.cross_entropy(outputs, y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Return updated model parameters and metrics
        return self.get_parameters({}), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        # Set model parameters
        self.set_parameters(parameters)
        
        # Evaluate model
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                total_loss += torch.nn.functional.cross_entropy(
                    outputs, y, reduction='sum'
                ).item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        
        # Return metrics
        loss = total_loss / total
        accuracy = correct / total
        
        return loss, len(self.val_loader.dataset), {
            'accuracy': accuracy,
            'loss': loss
        }

def start_client(model, train_loader, val_loader, device, server_address):
    """Start Flower client for federated learning."""
    # Create client
    client = AIClient(model, train_loader, val_loader, device)
    
    # Start client
    fl.client.start_numpy_client(server_address=server_address, client=client)
```

## Model Evaluation

### 1. Evaluation Metrics

```python
# training/metrics.py
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

def calculate_metrics(y_true, y_pred, y_proba=None, average='macro'):
    """Calculate various classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average),
        'recall': recall_score(y_true, y_pred, average=average),
        'f1': f1_score(y_true, y_pred, average=average),
    }
    
    if y_proba is not None and len(np.unique(y_true)) == 2:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
    
    return metrics

def print_confusion_matrix(y_true, y_pred, class_names=None):
    """Print confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]
    
    # Calculate row sums for percentages
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = cm / row_sums.astype(float) * 100
    
    print("Confusion Matrix (Counts / Percentages):")
    for i in range(len(cm)):
        print(f"{class_names[i] if i < len(class_names) else i}: ", end="")
        print(" ".join([f"{cm[i,j]:3d}({cm_percent[i,j]:.1f}%)" for j in range(len(cm[i]))]))
```

### 2. Evaluation Script

```python
# scripts/evaluate.py
import torch
from torch.utils.data import DataLoader
from models.cnn import SimpleCNN
from training.metrics import calculate_metrics, print_confusion_matrix

def evaluate_model(model, test_loader, device, class_names=None):
    """Evaluate model on test dataset."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probas = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            if outputs.size(1) == 2:  # Binary classification
                probas = torch.softmax(outputs, dim=1).cpu().numpy()
                all_probas.extend(probas)
    
    # Calculate metrics
    metrics = calculate_metrics(
        all_targets, 
        all_preds, 
        np.array(all_probas) if all_probas else None
    )
    
    # Print confusion matrix
    print_confusion_matrix(all_targets, all_preds, class_names)
    
    return metrics

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = SimpleCNN(num_classes=10)
    model.load_state_dict(torch.load('best_model.pth'))
    model = model.to(device)
    
    # Load test data
    _, test_loader = create_dataloaders(
        'data/', 
        batch_size=32, 
        test_split=0.2
    )
    
    # Evaluate model
    metrics = evaluate_model(
        model, 
        test_loader, 
        device,
        class_names=[str(i) for i in range(10)]
    )
    
    print("\nEvaluation Results:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

if __name__ == '__main__':
    main()
```

## Model Packaging

### 1. ONNX Export

```python
# scripts/export_onnx.py
import torch
from models.cnn import SimpleCNN

def export_onnx(model_path, output_path, input_shape=(1, 3, 224, 224)):
    """Export PyTorch model to ONNX format."""
    # Load model
    model = SimpleCNN(num_classes=10)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Export model
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {output_path}")

if __name__ == '__main__':
    export_onnx('best_model.pth', 'model.onnx')
```

### 2. Model Card

```yaml
# model_card.yaml
model:
  name: simple_cnn
  version: 1.0.0
  description: A simple CNN for image classification
  author: Your Name
  license: MIT

data:
  dataset: CIFAR-10
  input_shape: [224, 224, 3]
  num_classes: 10
  train_size: 40000
  val_size: 10000
  test_size: 10000

training:
  framework: PyTorch 1.9.0
  epochs: 50
  batch_size: 32
  optimizer: Adam
  learning_rate: 0.001
  loss: CrossEntropyLoss
  metrics: [accuracy, precision, recall, f1]

performance:
  accuracy: 0.9234
  precision: 0.9241
  recall: 0.9234
  f1: 0.9235
  inference_time: 0.015  # seconds per sample

usage: |
  ```python
  import torch
  from torchvision import transforms
  from PIL import Image
  
  # Load model
  model = torch.load('model.pth')
  model.eval()
  
  # Preprocess image
  transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
  ])
  
  # Make prediction
  image = Image.open('image.jpg')
  input_tensor = transform(image).unsqueeze(0)
  with torch.no_grad():
      output = model(input_tensor)
  predicted_class = output.argmax().item()
  ```

limitations:
  - Works best with images similar to CIFAR-10
  - May not generalize well to other domains

ethical_considerations:
  - Intended for research and educational purposes
  - Should not be used for critical applications without further testing
```

## Deployment

### 1. FastAPI Web Service

```python
# api/app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
import io
import numpy as np

app = FastAPI(title="AI Model API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = None
model_path = "best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Load model on startup
@app.on_event("startup")
async def load_model():
    global model
    model = SimpleCNN(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_prob, pred_class = torch.max(probs, 1)
        
        return {
            "class": int(pred_class.item()),
            "probability": float(pred_prob.item()),
            "all_probabilities": probs[0].cpu().numpy().tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. Dockerfile

```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the command to start the API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Monitoring and Maintenance

### 1. Logging

```python
# utils/logger.py
import logging
import os
from datetime import datetime

def setup_logger(name, log_dir='logs'):
    """Set up a logger with file and console handlers."""
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create file handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
```

### 2. Model Versioning

```python
# utils/model_registry.py
import os
import shutil
from datetime import datetime
import yaml

class ModelRegistry:
    def __init__(self, registry_dir='model_registry'):
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)
    
    def register_model(self, model_path, metadata):
        """Register a new model version in the registry."""
        # Generate version ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_dir = os.path.join(self.registry_dir, f'model_v{timestamp}')
        os.makedirs(version_dir, exist_ok=True)
        
        # Copy model file
        model_filename = os.path.basename(model_path)
        shutil.copy(model_path, os.path.join(version_dir, model_filename))
        
        # Save metadata
        metadata['timestamp'] = timestamp
        with open(os.path.join(version_dir, 'metadata.yaml'), 'w') as f:
            yaml.dump(metadata, f)
        
        # Update latest symlink
        latest_link = os.path.join(self.registry_dir, 'latest')
        if os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(version_dir, latest_link)
        
        return version_dir
    
    def get_latest_version(self):
        """Get the latest model version."""
        latest_link = os.path.join(self.registry_dir, 'latest')
        if not os.path.exists(latest_link):
            return None
        
        version_dir = os.path.realpath(latest_link)
        model_path = None
        metadata_path = os.path.join(version_dir, 'metadata.yaml')
        
        # Find model file (supports .pth, .pt, .h5, .onnx)
        for ext in ['.pth', '.pt', '.h5', '.onnx']:
            model_files = [f for f in os.listdir(version_dir) if f.endswith(ext)]
            if model_files:
                model_path = os.path.join(version_dir, model_files[0])
                break
        
        if not model_path:
            raise FileNotFoundError("No model file found in version directory")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        return {
            'version': os.path.basename(version_dir),
            'model_path': model_path,
            'metadata': metadata
        }
    
    def list_versions(self):
        """List all model versions in the registry."""
        versions = []
        for item in os.listdir(self.registry_dir):
            if item == 'latest':
                continue
            
            version_dir = os.path.join(self.registry_dir, item)
            if os.path.isdir(version_dir):
                versions.append(item)
        
        return sorted(versions, reverse=True)
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```python
# Reduce batch size
batch_size = 32  # Try reducing this

# Clear CUDA cache
torch.cuda.empty_cache()

# Use gradient accumulation
accumulation_steps = 4
optimizer.zero_grad()
for i, (inputs, targets) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss = loss / accumulation_steps  # Normalize loss
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 2. Model Not Learning
- Check learning rate (try values between 1e-5 and 1e-2)
- Verify data loading (are labels correct?)
- Check for vanishing/exploding gradients (use gradient clipping)
- Try a simpler model architecture first

#### 3. Overfitting
- Add more training data
- Use data augmentation
- Add dropout layers
- Increase weight decay (L2 regularization)
- Use early stopping

### Debugging

```python
# Print model summary
from torchsummary import summary
model = SimpleCNN(num_classes=10)
summary(model, (3, 224, 224))  # Input shape

# Check for NaN/Inf values
torch.autograd.set_detect_anomaly(True)
with torch.autograd.detect_anomaly():
    # Your training loop here
    pass

# Log gradients
def log_gradients(model, step, writer):
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f'grad/{name}', param.grad, step)
```

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Flower Federated Learning](https://flower.dev/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [MLflow](https://mlflow.org/) - For experiment tracking and model management
- [Weights & Biases](https://wandb.ai/) - For experiment tracking and visualization

## Getting Help

If you encounter any issues or have questions:

1. Check the [GitHub Issues](https://github.com/REChain-Network-Solutions/AIPlatform/issues)
2. Join our [Discord](https://discord.gg/aiplatform)
3. Ask on [Stack Overflow](https://stackoverflow.com/) with the `pytorch` or `tensorflow` tags
