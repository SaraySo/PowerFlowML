import tensorflow as tf

def build_lstm_model(input_shape, output_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, input_shape=input_shape),
        tf.keras.layers.Dense(output_shape)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm_model(model, X_train, y_train, X_test, y_test, epochs=50):
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    mse = model.evaluate(X_test, y_test)
    return model, mse
