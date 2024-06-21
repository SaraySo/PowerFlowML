import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_and_prepare_data(input_path, output_path, n_steps):
    df = pd.read_csv(input_path)
    df_target = pd.read_csv(output_path)

    # Check for missing values
    if df.isnull().any().any() or df_target.isnull().any().any():
        raise ValueError("Missing values found in the dataset")

    df_clean = df.copy()
    df_target_clean = df_target.copy()

    # Calculate correlation and drop highly correlated features
    features_corr = df_clean.corr().abs()
    upper_triangle = features_corr.where(np.triu(np.ones(features_corr.shape), k=1).astype(bool))
    thresh = 1  # 100% correlation
    drop_idx = [col for col in upper_triangle.columns if any(upper_triangle[col] >= thresh)]
    df_clean = df_clean.drop(drop_idx, axis=1)

    # Normalize input data
    scaler = MinMaxScaler()
    input_scaled = scaler.fit_transform(df_clean)
    output_scaled = scaler.fit_transform(df_target_clean)

    def create_dataset(X, y, n_steps):
        Xs, ys = [], []
        for i in range(len(X) - n_steps):
            Xs.append(X[i:(i + n_steps)])
            ys.append(y[i + n_steps])
        return np.array(Xs), np.array(ys)

    X, y = create_dataset(input_scaled, output_scaled, n_steps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # Reshape input for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], df_clean.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], df_clean.shape[1]))

    # Flatten the input for DNN
    X_train_flattened = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
    X_test_flattened = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

    return X_train, X_test, y_train, y_test, X_train_flattened, X_test_flattened, df_target_clean.shape[1]