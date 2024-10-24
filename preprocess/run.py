'''
Running script
'''

from preprocess.feature_extraction import extract
from preprocess.feature_separation import run_full

if __name__ == '__main__':
    extract()
    run_full()
