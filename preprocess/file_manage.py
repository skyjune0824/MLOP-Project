'''
File managing function will be here, collected.
'''

import pandas
import os

def load_pkl(file):
    return pandas.read_pickle(file)

def save_pkl(path, data):
    data.to_pickle(path)

def load_csv(file):
    return pandas.read_csv(file)

def save_csv(path, data):
    data.to_csv(path, index=False)
    
def get_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
