import argparse
import numpy as np
import plot_results
from train_model import run_model  # Ensure run_model is imported

def plot_results_main(y_test, y_pred_fnn=None, y_pred_lstm=None):
    plot_results.plot_predictions(y_test, y_pred_fnn, y_pred_lstm)
    plot_results.plot_accuracy(y_test, y_pred_fnn, y_pred_lstm)

def plot_loss(model_name):
    import matplotlib.pyplot as plt

    history = np.load(f"{model_name}_history.npy", allow_pickle=True).item()

    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run neural network training and evaluation.")
    parser.add_argument("task", type=str, help="The task to perform: 'train', 'plot', or 'plot_loss'")
    parser.add_argument("--models", type=str, nargs='+', help="The names of the models to run (e.g., 'FNN_2L_50N' 'LSTM_1L_50U')", required=False)
    args = parser.parse_args()

    if args.task == "train" and args.models:
        for model_name in args.models:
            y_test, y_pred, history = run_model(model_name)
            # Save predictions and true values for later plotting
            np.save(f"{model_name}_y_test.npy", y_test)
            np.save(f"{model_name}_y_pred.npy", y_pred)
    elif args.task == "plot" and args.models:
        y_test = np.load(f"{args.models[0]}_y_test.npy")
        y_pred_fnn = np.load(f"{args.models[0]}_y_pred.npy") if "fnn" in args.models[0].lower() else None
        y_pred_lstm = np.load(f"{args.models[0]}_y_pred.npy") if "lstm" in args.models[0].lower() else None
        if len(args.models) > 1:
            y_pred_fnn = np.load(f"{args.models[1]}_y_pred.npy") if "fnn" in args.models[1].lower() else y_pred_fnn
            y_pred_lstm = np.load(f"{args.models[1]}_y_pred.npy") if "lstm" in args.models[1].lower() else y_pred_lstm
        plot_results_main(y_test, y_pred_fnn, y_pred_lstm)
    elif args.task == "plot_loss" and args.models:
        plot_loss(args.models[0])
    else:
        print("Please provide a valid task and model names.")
