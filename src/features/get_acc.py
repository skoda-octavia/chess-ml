from torch.utils.data import DataLoader
import torch

def get_accuracy_class(model: torch.nn.Module, data_loader: DataLoader, device: torch.device):
    total_samples = 0
    correct = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct += torch.sum((labels[torch.arange(len(predicted)), predicted] == 1).float()).item()
                
    
    
    return correct / total_samples