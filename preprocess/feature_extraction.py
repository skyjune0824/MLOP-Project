'''
Extract relevant feature only, and save accordingly.
'''




from preprocess.file_manage import load_csv, get_files
from setting.paths import RAW_DIR

def extract():
    names = ['Traffic.csv']
    # names = ['Traffic.csv', 'TrafficTwoMonth.csv']
    
    for name in names:
        full_df = load_csv(name)
    
    # full_df = 
    