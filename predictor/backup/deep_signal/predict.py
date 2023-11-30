import numpy as np

from predictor.backup.deep_signal.model_setup import *


def predict_regress_model(curve_path, X_predict):
    # extract file name
    file_name = curve_path.split("\\")[-1].split(".")[0]

    X_predict = np.reshape(X_predict, (X_predict.shape[0], X_predict.shape[1], 1))

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the model_setup
    model = SimpleLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim)

    # Load the trained model_setup
    model.load_state_dict(torch.load('model_setup\{}_predict_model.pt'.format(file_name)))

    # Test the model_setup
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(X_predict).float().to(device)
        # Forward pass
        outputs = model(inputs)

    outputs = outputs.cpu().numpy()

    return outputs
