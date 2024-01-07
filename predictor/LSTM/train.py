# 导入必要的库
import torch.optim as optim

from predictor.LSTM.model_setup import *
from predictor.LSTM.data_loader import *
from predictor.LSTM.config import *

# 创建模型实例
model = LSTMModel(input_size, hidden_layer_size, output_size, num_layers)
criterion = nn.MSELoss()  # 例如，对于回归问题
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

train_loader, test_loader = build_lstm_stock_regression_data_set(feature_li, train_data_start_date, train_data_end_date,
                                                                 period_type, batch_size=64, data_filter=data_filter,
                                                                 sequence_length=20)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_train, y_train in train_loader:  # 假设train_loader是您的数据加载器
        X_train, y_train = X_train.to(device), y_train.to(device)
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(model.num_layers, 1, model.hidden_layer_size),
                             torch.zeros(model.num_layers, 1, model.hidden_layer_size))

        y_pred = model(X_train)
        single_loss = criterion(y_pred, y_train)
        single_loss.backward()
        optimizer.step()

        total_loss += single_loss.item()

    if (epoch + 1) % 10 == 0:
        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Train Loss: {average_loss}")

    # 预测
    model.eval()  # 将模型设置为评估模式
    total_test_loss = 0
    with torch.no_grad():
        for X_test, targets in test_loader:  # 假设test_loader是您的预测数据加载器
            y_test_pred = model(X_test)
            test_loss = criterion(y_test_pred, targets)
            total_test_loss += test_loss.item()

        if (epoch + 1) % 10 == 0:
            average_test_loss = total_test_loss / len(test_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Test Loss: {average_test_loss}")
            if average_test_loss < min_test_loss:
                min_test_loss = average_test_loss
                torch.save(model,
                           f'LSTM_reg_{train_data_start_date}_{train_data_end_date}-{period_type}-{data_filter}.pth')  # save the entire model


