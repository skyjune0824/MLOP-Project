'''
Convert features so that it could be inputted.

mostly string -> int
'''

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from preprocess.file_manage import load_csv, save_csv
from setting.paths import CSV_DIR


def convert():
    '''
    convert data types into numeric, so that the model could read it easily
    '''

    full_df = load_csv(f"{CSV_DIR}/full_features.csv")


    full_df['Time'] = pd.to_datetime(full_df['Time'], format='%I:%M:%S %p').dt.hour + \
                       pd.to_datetime(full_df['Time'], format='%I:%M:%S %p').dt.minute / 60

    full_df['Time'] = full_df['Date'] * 24 + full_df['Time']

    le = LabelEncoder()
    full_df['Day of the week'] = le.fit_transform(full_df['Day of the week'])

    save_csv(f"{CSV_DIR}/converted_features.csv", full_df)
