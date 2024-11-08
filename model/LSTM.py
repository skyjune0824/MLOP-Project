'''
LSTM model of traffic flow prediction
'''
import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

from preprocess.file_manage import load_csv
from setting.paths import TRAIN_DIR, TEST_DIR


def create_dataset(X, y, n_input=3):
    '''
    Create a dataset suitable for LSTM.

    Parameters:
        X (DataFrame): The input DataFrame.
        y (DataFrame): The target DataFrame.
        n_input (int): The number of time steps to look back.

    Returns:
        TimeseriesGenerator: A generator for training the model.
    '''
    generator = TimeseriesGenerator(X.values, y.values, length=n_input, batch_size=1)
    return generator


def build_lstm_model(input_shape):
    '''
    Build and compile the LSTM model.

    Parameters:
        input_shape (tuple): Shape of the input data.

    Returns:
        model: Compiled LSTM model.
    '''
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def main(hide = False):
    '''
    main running codes
    '''
    X_train = load_csv(f"{TRAIN_DIR}/X.csv")
    y_train = load_csv(f"{TRAIN_DIR}/y.csv")
    X_test = load_csv(f"{TEST_DIR}/X.csv")
    y_test = load_csv(f"{TEST_DIR}/y.csv")

    n_input = 3
    train_gen = create_dataset(X_train, y_train, n_input=n_input)
    test_gen = create_dataset(X_test, y_test, n_input=n_input)

    model = build_lstm_model(input_shape=(n_input, X_train.shape[1]))

    checkpoint = ModelCheckpoint('LSTM_best_model.keras', save_best_only=True, monitor='loss', mode='min')

    verbose = 2 if hide else 1

    model.fit(train_gen, epochs=200, verbose=verbose, callbacks=[checkpoint])

    loss = model.evaluate(test_gen)
    print(f'Test Loss: {loss}')


if __name__ == "__main__":

    hide = '-h' in sys.argv or '--hide' in sys.argv

    main(hide=hide)
