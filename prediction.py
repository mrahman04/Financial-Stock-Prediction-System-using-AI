import numpy as np

def make_predictions(model, X_test, scaler):
    """
    Make predictions using the trained model and inverse transform them to original scale.

    Args:
        model (tensorflow.keras.Model): Trained LSTM model.
        X_test (np.ndarray): Test input data.
        scaler (MinMaxScaler): Scaler used during data preprocessing.

    Returns:
        np.ndarray: Predicted stock prices in the original scale.
    """
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions

def evaluate_model(predictions, y_test, scaler):
    """
    Evaluate model performance by comparing predictions with actual values.

    Args:
        predictions (np.ndarray): Predicted stock prices.
        y_test (np.ndarray): Actual stock prices in normalized scale.
        scaler (MinMaxScaler): Scaler used during data preprocessing.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    # Convert y_test back to original scale
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate evaluation metrics
    mae = np.mean(np.abs(predictions - y_test_original))
    mse = np.mean((predictions - y_test_original) ** 2)
    rmse = np.sqrt(mse)

    return {"MAE": mae, "MSE": mse, "RMSE": rmse}
