import tensorflow as tf
from loss_functions import get_loss_function
import numpy as np

def compile_and_train_model(model, X_train, y_train, X_test, y_test, config):
    loss_function = get_loss_function(config['loss_function'])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']), loss=loss_function)

    history = model.fit(X_train, y_train, epochs=config['epochs'], validation_data=(X_test, y_test))
    mse = model.evaluate(X_test, y_test)
    model.save(f"{config['model_name']}_model.h5")  # Save the model
    np.save(f"{config['model_name']}_history.npy", history.history)  # Save the training history

    return model, mse, history.history  # Return history
