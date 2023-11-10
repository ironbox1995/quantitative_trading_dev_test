import matplotlib.pyplot as plt

from deep_signal.pick_time_utils import *
from deep_signal.model import *
from deep_signal.data_loader import *


def train_lstm_regress_model(curve_path, X_train, y_train, l2_reg=L2_REG_COEFFICIENT):
    # extract file name
    file_name = curve_path.split("\\")[-1].split(".")[0]

    split_idx = int(0.9 * len(X_train))
    X_train, X_val = X_train[:split_idx], X_train[split_idx:]
    y_train, y_val = y_train[:split_idx], y_train[split_idx:]

    # Reshape the data
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if ATTENTION:
        model = LSTMAttentionModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYER, output_size=1, device=device).to(device)
    else:
        model = LSTMModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYER, output_size=1, dropout_rate=DROPOUT, device=device).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=l2_reg)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Create train loader
    train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # Create validation loader
    val_loader = create_val_loader(X_val, y_val)

    # Train the model
    num_epochs = NUM_EPOCH
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()  # pytorch中，训练时要转换成训练模式
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item() * inputs.size(0)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader.dataset)
        if (epoch + 1) % 10 == 0:
            val_loss = evaluate_regress_model(model, val_loader, criterion, device)  # evaluate the model on the validation set
            if train_loss < best_loss:  # if the train loss is better than the current best loss
                best_loss = train_loss  # update the best loss
                # Save the model as a pt file
                torch.save(model.state_dict(), 'model\{}_predict_model.pt'.format(file_name))
                print(f"Model saved successfully at epoch {epoch + 1} with train loss {best_loss}")

            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.14f}, Validation Loss: {val_loss:.14f}')
        if np.isnan(train_loss):
            print(f"Training stopped at epoch {epoch + 1} due to NaN loss")
            break


if __name__ == "__main__":
    # 两个函数输入的起止日期一定要一样，因为是在各自内部划分训练和测试数据
    strategy_name = "小市值策略"
    period_type = "W"
    select_stock_num = 3
    date_start = '2010-01-01'
    date_end = '2023-03-31'
    pick_time_mtd = "无择时"
    curve_path = r"F:\quantitative_trading_dev_test\quant_test\backtest\result_record\select_stock_{}_{}_选{}_{}-{}_{}.csv"\
        .format(strategy_name, period_type, select_stock_num, date_start, date_end, pick_time_mtd)
    X_train, y_train, X_test, y_test = data_split(curve_path, date_start, date_end, CLASSIFY=False)
    train_lstm_regress_model(curve_path, X_train, y_train)
    test_regress_model(curve_path, X_test, y_test)



