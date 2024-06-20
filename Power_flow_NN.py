import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import model_selection, preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# import data
df = pd.read_csv(r'Data\in_data_1752.csv')
df_target = pd.read_csv(r'Data\out_data_1752.csv')

# get the data sizes
print(f'input shape:  {df.shape}')
print(f'output shape:  {df_target.shape}')

# check for missing values
df.isnull().any().any()

# Creating a clean data values:
df_clean = df.copy()
df_target_clean = df_target.copy()

# calculate correlation between feature columns, we take the absolute value (size of correlation)
features_corr = df_clean.corr().abs()

# refer only to the uupper triangle of the corr matrix:
upper_triangle = features_corr.where(np.triu(np.ones(features_corr.shape), k=1).astype(bool))

# drop columns above the correlation threshold
thresh = 1 # 100% correlation
drop_idx = [col for col in upper_triangle.columns if any(upper_triangle[col] >= thresh)]

# how many columns to drop?
print(f'drop: {len(drop_idx)} columns')

# drop columns
df_clean = df_clean.drop(drop_idx, axis = 1)
print(f'remaining data:  {df_clean.shape}')

# Normalize input data
scaler = MinMaxScaler()
input_scaled = scaler.fit_transform(df_clean)
output_scaled = scaler.fit_transform(df_target_clean)

# Define a function to create dataset for LSTM and DNN
def create_dataset(X, y, n_steps):
    Xs, ys = [], []
    for i in range(len(X) - n_steps):
        Xs.append(X[i:(i + n_steps)])
        ys.append(y[i + n_steps])
    return np.array(Xs), np.array(ys)

n_steps = 5  # Number of steps
X, y = create_dataset(input_scaled, output_scaled, n_steps)

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Reshape input for LSTM [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], df_clean.shape[1]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], df_clean.shape[1]))

#DNN model
model_dnn = tf.keras.models.Sequential([
    tf.keras.layers.Dense(50, activation='relu', input_shape=(X_train.shape[1]*X_train.shape[2],)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(df_target_clean.shape[1])
])
model_dnn.compile(optimizer='adam', loss='mse')
# Flatten the input for DNN because it does not accept 3D data [samples, timesteps, features]
X_train_flattened = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
X_test_flattened = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))

# Train the DNN model
history_dnn = model_dnn.fit(X_train_flattened, y_train, epochs=50, validation_data=(X_test_flattened, y_test))

#LSTM model
model_lstm = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Dense(df_target_clean.shape[1])
])
model_lstm.compile(optimizer='adam', loss='mse')

# Evaluation
mse_lstm = model_lstm.evaluate(X_test, y_test)
mse_dnn = model_dnn.evaluate(X_test_flattened, y_test)

print("MSE for LSTM:", mse_lstm)
print("MSE for DNN:", mse_dnn)

# Predicting with models
y_pred_lstm = model_lstm.predict(X_test)
y_pred_dnn = model_dnn.predict(X_test_flattened)

# Plotting the results
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

# Scatter plot to show accuracy of predictions
plt.figure(figsize=(12, 6))

# LSTM Prediction Accuracy
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lstm, alpha=0.5, color='red')
plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'k--', lw=2)  # Diagonal line
plt.title('LSTM Prediction Accuracy')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# DNN Prediction Accuracy
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_dnn, alpha=0.5, color='green')
plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'k--', lw=2)  # Diagonal line
plt.title('DNN Prediction Accuracy')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.tight_layout()
plt.show()