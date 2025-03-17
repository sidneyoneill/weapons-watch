import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class ArmsTradeCNN(nn.Module):
    def __init__(self):
        super(ArmsTradeCNN, self).__init__()
        
        # CNN1 to CNN6 layers
        self.cnn_layers = nn.Sequential(
            # CNN1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            # CNN2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            # CNN3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            # CNN4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            
            # CNN5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            
            # CNN6
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 14),  # Output size: 14 as specified
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.fc(x)
        return x

class ArmsTradeDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_model(model, train_loader, valid_loader, num_epochs=100, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_valid_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        valid_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in valid_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                valid_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        # Calculate average losses and accuracy
        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)
        accuracy = 100 * correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Valid Loss: {avg_valid_loss:.4f}')
        print(f'Accuracy: {accuracy:.2f}%')
        
        # Save the best model
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), 'best_model.pth')

def prepare_data(data_path, sequence_length=1024):
    """
    Prepare the data for training. This function needs to be customized based on your data format.
    """
    # Load your filtered data
    df = pd.read_csv(data_path)
    
    # Implement your data preprocessing here
    # This is a placeholder - you'll need to modify this based on your actual data structure
    
    return data, labels

def main():
    # Initialize the model
    model = ArmsTradeCNN()
    
    # Load and prepare your data
    # You'll need to implement the data loading and preprocessing
    data_path = 'path_to_your_filtered_data.csv'
    data, labels = prepare_data(data_path)
    
    # Create train/validation split
    from sklearn.model_selection import train_test_split
    train_data, valid_data, train_labels, valid_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )
    
    # Create data loaders
    train_dataset = ArmsTradeDataset(train_data, train_labels)
    valid_dataset = ArmsTradeDataset(valid_data, valid_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32)
    
    # Train the model
    train_model(model, train_loader, valid_loader)

if __name__ == "__main__":
    main()