'''
Split train/test data, as well as separate out the prediction objectives
'''

from sklearn.model_selection import train_test_split

from preprocess.file_manage import load_csv, save_csv
from setting.paths import CSV_DIR, TEST_DIR, TRAIN_DIR


def separate(df_path, target="Traffic Situation"):
    '''
    Extract y
    '''

    df = load_csv(df_path)

    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def split(X, y, test_size=0.2):
    '''
    Train/Test split
    '''

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test


def run_full():
    '''
    run above functions
    '''

    X, y = separate(f"{CSV_DIR}/converted_features.csv", target="Traffic Situation")
    X_train, X_test, y_train, y_test = split(X, y)

    save_csv(f"{TRAIN_DIR}/X.csv", X_train)
    save_csv(f"{TRAIN_DIR}/y.csv", y_train)
    save_csv(f"{TEST_DIR}/X.csv", X_test)
    save_csv(f"{TEST_DIR}/y.csv", y_test)
