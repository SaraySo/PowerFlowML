import json
import data_preparation
import dnn_model
import lstm_model
import matplotlib.pyplot as plt
import numpy as np


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config


def main():
    config = load_config('config.json')

    X_train, X_test, y_train, y_test, X_train_flattened, X_test_flattened, output_shape = data_preparation.load_and_prepare_data(
        config['input_path'], config['output_path'], config['n_steps'])

    # Train DNN
    dnn = dnn_model.build_dnn_model(X_train_flattened.shape[1], output_shape)
    dnn, mse_dnn = dnn_model.train_dnn_model(dnn, X_train_flattened, y_train, X_test_flattened, y_test,
                                             config['epochs'])

    # Train LSTM
    lstm = lstm_model.build_lstm_model((X_train.shape[1], X_train.shape[2]), output_shape)
    lstm, mse_lstm = lstm_model.train_lstm_model(lstm, X_train, y_train, X_test, y_test, config['epochs'])

    print("MSE for LSTM:", mse_lstm)
    print("MSE for DNN:", mse_dnn)

    # Predict and plot
    y_pred_lstm = lstm.predict(X_test)
    y_pred_dnn = dnn.predict(X_test_flattened)

    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.plot(y_test)
    plt.plot(y_pred_lstm)
    plt.title('Comparison of LSTM Predictions and Actual Data')
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Values')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(y_test)
    plt.plot(y_pred_dnn)
    plt.title('Comparison of DNN Predictions and Actual Data')
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Values')
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_lstm, alpha=0.5, color='red')
    plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'k--', lw=2)
    plt.title('LSTM Prediction Accuracy')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_dnn, alpha=0.5, color='green')
    plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'k--', lw=2)
    plt.title('DNN Prediction Accuracy')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
