from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVR, SVC

from machine_learning.data_loader import *
from machine_learning.nn_model import *
from machine_learning.model_config import *
from machine_learning.ml_utils import *


def train_random_forest_classify_model(start_date, end_date, data_type, n_estimators=100):
    X_train, y_train, X_test, y_test = build_classify_data_set(start_date, end_date, data_type)

    # Remove the random_state parameter to generate actual predictions
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    clf.fit(X_train, y_train)

    with open('model_setup/random_forest_classify_model_{}_{}-{}.pkl'.format(data_type, start_date, end_date), 'wb') as f:
        pickle.dump(clf, f)


def train_SVC_model(start_date, end_date, data_type):
    X_train, y_train, X_test, y_test = build_classify_data_set(start_date, end_date, data_type)

    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)

    with open('model_setup/SVC_classify_model_{}_{}-{}.pkl'.format(data_type, start_date, end_date), 'wb') as f:
        pickle.dump(clf, f)


def train_random_forest_regress_model(start_date, end_date, data_type, n_estimators=100):
    X_train, y_train, X_test, y_test = build_regression_data_set(start_date, end_date, data_type)

    # Remove the random_state parameter to generate actual predictions
    clf = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
    clf.fit(X_train, y_train)

    with open('model_setup/random_forest_regress_model_{}_{}-{}.pkl'.format(data_type, start_date, end_date), 'wb') as f:
        pickle.dump(clf, f)


def train_SVR_model(start_date, end_date, data_type):
    X_train, y_train, X_test, y_test = build_regression_data_set(start_date, end_date, data_type)

    clf = SVR()
    clf.fit(X_train, y_train)

    with open('model_setup/SVR_regress_model_{}_{}-{}.pkl'.format(data_type, start_date, end_date), 'wb') as f:
        pickle.dump(clf, f)


def train_FCN_classify_model(start_date, end_date, data_type, X_train, y_train, patience, hidden_size=50, num_hidden_layers=2, batch_size=64, lr=clf_lr, l2_reg=l2_reg_coefficient):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # X_train, y_train, X_test, y_test = build_classify_data_set(start_date, end_date, data_type)

    split_idx = int(0.9 * len(X_train))
    X_train, X_val = X_train[:split_idx], X_train[split_idx:]
    y_train, y_val = y_train[:split_idx], y_train[split_idx:]

    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    model = FCN(input_size, hidden_size, num_classes, num_hidden_layers)
    model.to(device)
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0])).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Convert data to PyTorch tensors
    train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Create validation loader
    val_loader = create_clf_val_loader(X_val, y_val)

    # Train the model_setup
    num_epochs = total_epoch
    best_loss = np.inf
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)

        if (epoch+1) % 10 == 0:
            val_loss = evaluate_classify_model(model, val_loader, criterion, device)  # evaluate the model_setup on the validation set
            print('Epoch [{}/{}], Train Loss: {:.14f}, Val Loss: {:.14f}'.format(epoch+1, num_epochs, train_loss, val_loss))

            if train_loss < best_loss:
                best_loss = train_loss
                # Save the model_setup
                torch.save(model.state_dict(), 'model_setup/FCN_classify_model_{}_{}-{}.pt'.format(data_type, start_date, end_date))
                print(f"Model saved successfully at epoch {epoch + 1} with train loss {best_loss}")
        #     else:
        #         patience -= 10
        #
        # if patience <= 0:
        #     print("Early stopping at epoch {}".format(epoch))
        #     break


def train_FCN_regress_model(start_date, end_date, data_type, X_train, y_train, patience, hidden_size=50
                            , num_hidden_layers=2, batch_size=64, lr=rgs_lr, l2_reg=l2_reg_coefficient):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # X_train, y_train, X_test, y_test = build_regression_data_set(start_date, end_date, data_type)

    split_idx = int(0.9 * len(X_train))
    X_train, X_val = X_train[:split_idx], X_train[split_idx:]
    y_train, y_val = y_train[:split_idx], y_train[split_idx:]

    input_size = X_train.shape[1]
    output_size = 1

    model = FCN(input_size, hidden_size, output_size, num_hidden_layers)
    model.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)

    # Create train loader
    train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Create validation loader
    val_loader = create_rgs_val_loader(X_val, y_val)

    # Train the model_setup
    num_epochs = total_epoch
    best_loss = np.inf
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)

        if (epoch+1) % 10 == 0:
            val_loss = evaluate_regress_model(model, val_loader, criterion, device)  # evaluate the model_setup on the validation set
            print('Epoch [{}/{}], Train Loss: {:.14f}, Val Loss: {:.14f}'.format(epoch+1, num_epochs, train_loss, val_loss))

            if train_loss < best_loss:
                best_loss = train_loss
                # Save the model_setup
                torch.save(model.state_dict(), 'model_setup/FCN_regress_model_{}_{}-{}.pt'.format(data_type, start_date, end_date))
                print(f"Model saved successfully at epoch {epoch + 1} with train loss {best_loss}")
        #     else:
        #         patience -= 10
        #
        # if patience <= 0:
        #     print("Early stopping at epoch {}".format(epoch))
        #     break
