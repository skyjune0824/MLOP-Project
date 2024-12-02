'''
transformer model of traffic flow prediction
'''
import sys
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from preprocess.file_manage import load_csv
from setting.paths import TRAIN_DIR, TEST_DIR
warnings.filterwarnings('ignore')


class TrafficTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads=7, n_layers=6, dropout=0.1):
        super(TrafficTransformer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout

        self.positional_encoding = nn.Parameter(
            torch.rand(n_layers, input_dim), requires_grad=True)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=n_heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=n_layers)

        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")

        if len(x.shape) == 2:
            seq_len = 1  # If the input is 2D, treat it as having a sequence length of 1
            x = x.unsqueeze(1)  # Reshape to (batch_size, seq_len, input_dim)
        else:
            seq_len = x.size(1)

        x = x.transpose(0, 1)

        seq_len = x.size(0)
        x = x + self.positional_encoding[:seq_len, :].unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)

        # Output layer
        x = self.fc(x)

        return x


def create_dataset(X, y, n_input=3, batch_size=1):
    '''
    Create a dataset suitable for LSTM.

    Parameters:
        X (DataFrame): The input.
        y (DataFrame): The target.
        n_input (int): The number of time steps to look back.

    Returns:
        TimeseriesGenerator: A generator for training the model.
    '''
    generator = TimeseriesGenerator(
        X.values, y.values, length=n_input, batch_size=batch_size)
    
    # for X_batch, y_batch in generator:
    #     print(f"X_batch shape: {X_batch.shape}")
    #     print(f"y_batch shape: {y_batch.shape}")
    #     break

    return generator


def main(hide=False):
    '''
    main running codes
    '''
    X_train = load_csv(f"{TRAIN_DIR}/X.csv")
    y_train = load_csv(f"{TRAIN_DIR}/y.csv")
    X_test = load_csv(f"{TEST_DIR}/X.csv")
    y_test = load_csv(f"{TEST_DIR}/y.csv")

    n_input = 5
    batch_size = 64


    train_gen = create_dataset(
        X_train, y_train, n_input=n_input, batch_size=batch_size)
    test_gen = create_dataset(
        X_test, y_test, n_input=n_input, batch_size=batch_size)

    input_dim = X_train.shape[1]

    model = TrafficTransformer(
        input_dim=input_dim, output_dim=len(np.unique(y_train)))

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 1
    for epoch in range(epochs):
        model.train()  # training mode
        running_loss = 0.0
        for X_batch, y_batch in tqdm(train_gen):
            if X_batch.shape[0] == 0:  # why there's empty batches??? why??????
                break
            X_batch = torch.tensor(X_batch, dtype=torch.float32)
            y_batch = torch.tensor(y_batch, dtype=torch.long)

            optimizer.zero_grad()

            outputs = model(X_batch)
            outputs_last = outputs[:, -1, :]

            loss = criterion(outputs_last, y_batch.view(-1))
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        # Print average loss for the epoch
        print(
            f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_gen)}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_gen:
            if X_batch.shape[0] == 0:  # just in case
                break
            X_batch = torch.tensor(X_batch, dtype=torch.float32)
            y_batch = torch.tensor(y_batch, dtype=torch.long)

            outputs = model(X_batch)
            outputs_last = outputs[:, -1, :]

            print(f"Raw outputs: {outputs_last}")
            _, predicted = torch.max(outputs_last, 1)

            print(f"Predicted: {predicted}, Actual: {y_batch.squeeze()}")

            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    print(f"Test Accuracy: {100 * correct / total}%")


if __name__ == "__main__":

    hide = '-h' in sys.argv or '--hide' in sys.argv

    main(hide=hide)
