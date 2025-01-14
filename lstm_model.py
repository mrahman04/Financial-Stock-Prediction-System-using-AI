import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(sequence_length, dropout_rate=0.2):
    """
    Build an LSTM model for stock price prediction.

    Args:
        sequence_length (int): Length of input sequences.
        dropout_rate (float): Dropout rate to prevent overfitting.

    Returns:
        tensorflow.keras.Model: Compiled LSTM model.
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(dropout_rate),
        LSTM(50, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model
