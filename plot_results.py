import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(y_test, y_pred_fnn=None, y_pred_lstm=None):
    if y_pred_fnn is not None:
        plt.figure(figsize=(14, 7))
        plt.plot(y_test, label='True')
        plt.plot(y_pred_fnn, label='Predicted')
        plt.title('Comparison of FNN Predictions and Actual Data')
        plt.xlabel('Time Steps')
        plt.ylabel('Normalized Values')
        plt.legend(['True', 'Predicted'])
        plt.show()

    if y_pred_lstm is not None:
        plt.figure(figsize=(14, 7))
        plt.plot(y_test, label='True')
        plt.plot(y_pred_lstm, label='Predicted')
        plt.title('Comparison of LSTM Predictions and Actual Data')
        plt.xlabel('Time Steps')
        plt.ylabel('Normalized Values')
        plt.legend(['True', 'Predicted'])
        plt.show()

def plot_accuracy(y_test, y_pred_fnn=None, y_pred_lstm=None):
    if y_pred_fnn is not None:
        plt.figure(figsize=(12, 6))
        plt.scatter(y_test, y_pred_fnn, alpha=0.5, color='green')
        plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'k--', lw=2)
        plt.title('FNN Prediction Accuracy')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.show()

    if y_pred_lstm is not None:
        plt.figure(figsize=(12, 6))
        plt.scatter(y_test, y_pred_lstm, alpha=0.5, color='red')
        plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'k--', lw=2)
        plt.title('LSTM Prediction Accuracy')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.show()
