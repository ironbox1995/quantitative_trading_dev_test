import numpy as np

from deep_signal.pick_time_utils import *


def predict_regress_model(curve_path, X_predict):
    # extract file name
    file_name = curve_path.split("\\")[-1].split(".")[0]

    X_predict = np.reshape(X_predict, (X_predict.shape[0], X_predict.shape[1], 1))

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if ATTENTION:
        model = LSTMAttentionModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYER, output_size=1,
                                   device=device).to(device)
    else:
        model = LSTMModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYER, output_size=1,
                          dropout_rate=DROPOUT, device=device).to(device)

    # Load the trained model
    model.load_state_dict(torch.load('model\{}_predict_model.pt'.format(file_name)))

    # Test the model
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(X_predict).float().to(device)
        # Forward pass
        outputs = model(inputs)

    outputs = outputs.cpu().numpy()
    # outputs = outputs[1:]  # cheat

    # calculate the accuracy rate of trend prediction
    if DIFF:
        outputs_trend = []
        for i in range(len(outputs)):
            if outputs[i] > 0:
                outputs_trend.append(1)
            else:
                outputs_trend.append(0)
    else:
        outputs_trend = positive_differences(outputs)

    return outputs_trend, outputs