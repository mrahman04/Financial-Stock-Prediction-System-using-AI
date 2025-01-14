import numpy as np
from sklearn.model_selection import train_test_split
from data_preprocessing import load_and_preprocess_data, create_sequences
from lstm_model import build_lstm_model

def train_model(data_path, sequence_length, epochs, batch_size):
    """
    Train the LSTM model using preprocessed data.

    Args:
        data_path (str): Path to the historical stock data CSV file.
        sequence_length (int): Number of time steps in each input sequence.
        epochs (int): Number of training epochs.
        batch_size (int): Size of training batches.

    Returns:
        tuple: Trained model and test data (X_test, y_test).
    """
    # Load and preprocess the data
    data, scaler = load_and_preprocess_data(data_path)
    sequences, labels = create_sequences(data, sequence_length)

    # Convert to numpy arrays
    X = np.array(sequences)
    y = np.array(labels)

    # Reshape input data for LSTM
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the LSTM model
    model = build_lstm_model(sequence_length)

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    return model, X_test, y_test
