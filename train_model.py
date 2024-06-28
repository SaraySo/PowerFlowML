import json
import numpy as np
import importlib
import data_preparation
from train import compile_and_train_model

def load_config(config_path, model_name):
    with open(config_path, 'r') as file:
        config = json.load(file)
    config[model_name]['model_name'] = model_name  # Add model name to config
    return config

def train_model(model_name, config):
    # Load and prepare data
    X_train, X_test, y_train, y_test, X_train_flattened, X_test_flattened, output_shape = data_preparation.load_and_prepare_data(
        config[model_name]['input_path'], config[model_name]['output_path'], config[model_name]['n_steps'])

    # Import model dynamically
    model_module = importlib.import_module(f"models.{model_name.lower()}")
    model_build_function = getattr(model_module, f"build_{model_name.lower()}_model")

    # Build the model
    if "fnn" in model_name.lower():
        model = model_build_function(X_train_flattened.shape[1], output_shape, config[model_name])
        X_train_use, X_test_use = X_train_flattened, X_test_flattened
    else:
        model = model_build_function((X_train.shape[1], X_train.shape[2]), output_shape, config[model_name])
        X_train_use, X_test_use = X_train, X_test

    # Train the model
    model, mse, history = compile_and_train_model(model, X_train_use, y_train, X_test_use, y_test, config[model_name])

    print(f"MSE for {model_name}:", mse)

    return model, X_test_use, y_test, history

def run_model(model_name):
    config = load_config('config.json', model_name)
    model, X_test_use, y_test, history = train_model(model_name, config)
    y_pred = model.predict(X_test_use)

    return y_test, y_pred, history
