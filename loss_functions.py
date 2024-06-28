import tensorflow as tf

def get_loss_function(loss_name):
    if loss_name == "mse":
        return tf.keras.losses.MeanSquaredError()
    elif loss_name == "mae":
        return tf.keras.losses.MeanAbsoluteError()
    elif loss_name == "huber":
        return tf.keras.losses.Huber()
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")
