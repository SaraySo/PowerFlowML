import tensorflow as tf

def build_fnn_model(input_shape, output_shape, config):
    model = tf.keras.models.Sequential()
    for units in config['hidden_layers']:
        model.add(tf.keras.layers.Dense(units, activation=config['activation'], input_shape=(input_shape,)))
        input_shape = None  # Only the first layer needs the input shape
    model.add(tf.keras.layers.Dense(output_shape))
    return model
