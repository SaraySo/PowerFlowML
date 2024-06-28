import tensorflow as tf

def build_lstm_1l_50u_model(input_shape, output_shape, config):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(config['units'], input_shape=input_shape),
        tf.keras.layers.Dense(output_shape)
    ])
    return model
