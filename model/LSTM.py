'''
LSTM model of traffic flow prediction
'''
import sys
import numpy
import warnings
import pandas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from preprocess.file_manage import load_csv
from setting.paths import TRAIN_DIR, TEST_DIR
import tensorflow as tf
from imblearn.over_sampling import SMOTE
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')


def create_dataset(X, y, n_input=3, batch_size=1):
    '''
    Create a dataset suitable for LSTM.

    Parameters:
        X (DataFrame): The input DataFrame.
        y (DataFrame): The target DataFrame.
        n_input (int): The number of time steps to look back.

    Returns:
        TimeseriesGenerator: A generator for training the model.
    '''
    generator = TimeseriesGenerator(
        X.values, y.values, length=n_input, batch_size=batch_size)
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
    model.add(Input(shape=input_shape))
    model.add(LSTM(128, activation='relu',
              return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='softmax'))  # 0 ~ 3, low-normal-high-heavy
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


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

    X_train_reshaped = pandas.DataFrame(X_train.to_numpy().reshape(X_train.shape[0], -1))

    original_size = {0: 686, 1: 2887, 2: 287, 3: 901}
    smote = SMOTE(sampling_strategy={0: 1500, 1: 2887, 2: 1000, 3: 2000}, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_reshaped, y_train)

    train_gen = create_dataset(
        X_resampled, y_resampled, n_input=n_input, batch_size=batch_size)
    test_gen = create_dataset(
        X_test, y_test, n_input=n_input, batch_size=batch_size)

    model = build_lstm_model(input_shape=(n_input, X_train.shape[1]))

    checkpoint = ModelCheckpoint(
        'LSTM_best_model.keras', save_best_only=True, monitor='loss', mode='min')
    # early_stopping = EarlyStopping(
    #     monitor='val_loss', patience=10, restore_best_weights=True)
    # No validation split for timeseries

    verbose = 2 if hide else 1

    history = model.fit(train_gen, epochs=300, batch_size=batch_size,
                        verbose=verbose, callbacks=[checkpoint])

    loss, acc = model.evaluate(test_gen)
    print(f'Test Loss: {loss}, Test Accuracy: {acc}')

if __name__ == "__main__":

    hide = '-h' in sys.argv or '--hide' in sys.argv

    main(hide=hide)
