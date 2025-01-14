import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path):
    """
    Load historical stock data and preprocess it for model training.

    Args:
        file_path (str): Path to the CSV file containing historical stock data.

    Returns:
        tuple: Normalized data, scaler instance, and processed DataFrame.
    """
    # Load data
    data = pd.read_csv(file_path)
    
    # Ensure necessary columns exist
    if 'Close' not in data.columns:
        raise ValueError("CSV file must contain 'Close' column.")

    # Normalize the 'Close' column
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['Normalized_Close'] = scaler.fit_transform(data[['Close']])

    return data, scaler

def create_sequences(data, sequence_length):
    """
    Create sequences for LSTM training.

    Args:
        data (pd.DataFrame): Preprocessed DataFrame.
        sequence_length (int): Number of time steps in each sequence.

    Returns:
        tuple: Numpy arrays of input sequences and corresponding labels.
    """
    sequences = []
    labels = []

    for i in range(len(data) - sequence_length):
        seq = data['Normalized_Close'].iloc[i:i + sequence_length].values
        label = data['Normalized_Close'].iloc[i + sequence_length]
        sequences.append(seq)
        labels.append(label)

    return sequences, labels
