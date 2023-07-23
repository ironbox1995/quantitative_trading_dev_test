from torch.utils.data import DataLoader, TensorDataset

from deep_signal.model import *
from deep_signal.pick_time_model_config import *


# Define a function to create the validation loader
def create_val_loader(X_val, y_val):
    val_data = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    return val_loader


# Define a function to evaluate the model on the validation set
def evaluate_regress_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(val_loader.dataset)
    return val_loss


# Define a function to evaluate the model on the validation set
def evaluate_classify_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    return val_loss/len(val_loader)


def adjacent_difference(lst, k=1):
    return [lst[i+k]-lst[i] for i in range(len(lst)-k)]


def positive_differences(lst, k=1):
    return [1 if diff > BUY_STANDARD else 0 for diff in adjacent_difference(lst, k)]