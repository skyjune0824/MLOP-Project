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

    max_date = 0

    date_col = full_df['Date'].tolist()

    for i in range(1, len(date_col)):

        date_col[i] += max_date
        if date_col[i] < date_col[i-1]:
            max_date = date_col[i-1]

    full_df['Date'] = date_col

    full_df['Time'] = full_df['Date'] * 24 + full_df['Time']
    full_df = full_df.drop(columns=['Date'])

    le = LabelEncoder()
    full_df['Day of the week'] = le.fit_transform(full_df['Day of the week'])

    traffic_mapping = {'low': 0, 'normal': 1, 'high': 2, 'heavy': 3}
    full_df['Traffic Situation'] = full_df['Traffic Situation'].map(traffic_mapping)

    save_csv(f"{CSV_DIR}/converted_features.csv", full_df)
