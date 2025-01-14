import matplotlib.pyplot as plt

def plot_predictions(actual, predicted):
    """
    Plot actual vs. predicted stock prices.

    Args:
        actual (np.ndarray): Actual stock prices.
        predicted (np.ndarray): Predicted stock prices.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual Prices', color='blue')
    plt.plot(predicted, label='Predicted Prices', color='red')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def save_model(model, file_path):
    """
    Save the trained model to a file.

    Args:
        model (tensorflow.keras.Model): Trained LSTM model.
        file_path (str): File path to save the model.
    """
    model.save(file_path)

def load_model(file_path):
    """
    Load a trained model from a file.

    Args:
        file_path (str): Path to the saved model file.

    Returns:
        tensorflow.keras.Model: Loaded LSTM model.
    """
    from tensorflow.keras.models import load_model
    return load_model(file_path)
