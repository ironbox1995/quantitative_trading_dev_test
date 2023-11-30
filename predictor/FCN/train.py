from model_setup import *
from data_loader import *
import torch.optim as optim
from config import *


def train(feature_li, data_start_date, data_end_date, epochs):
    """
    train FCN regress model
    """

    # loader
    train_loader, test_loader = build_stock_regression_data_set(feature_li, data_start_date, data_end_date, period_type, 64)

    # Create the model instance
    model = FullyConnectedRegressionNetwork(len(feature_li), hidden_sizes=[128, 64, 32], dropout_prob=0.5)
    # Loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # initialize min_test_loss
    min_test_loss = float("inf")

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        # training
        for inputs, targets in train_loader:  # train on each mini-batch
            optimizer.zero_grad()  # Clear the gradients

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            average_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Average Train Loss: {average_loss}")

        # Evaluation on test data
        model.eval()  # Set the model to evaluation mode
        total_test_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                predictions = model(inputs)
                test_loss = criterion(predictions, targets)
                total_test_loss += test_loss.item()

        if (epoch + 1) % 10 == 0:
            average_test_loss = total_test_loss / len(test_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Average Test Loss: {average_test_loss}")
            if average_test_loss < min_test_loss:
                min_test_loss = average_test_loss
                torch.save(model, f'FCN_reg_{data_start_date}_{data_end_date}.pth')  # save the entire model


if __name__ == "__main__":
    epochs = 3000
    train(feature_li, data_start_date, data_end_date, epochs)