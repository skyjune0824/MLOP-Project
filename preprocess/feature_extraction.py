'''
Extract relevant feature only, and save accordingly.
'''


from preprocess.file_manage import load_csv, get_files, save_csv
from setting.paths import RAW_DIR, CSV_DIR


def extract():
    '''
    Extract relavent feature only and save it.
    '''

    names = ['Traffic.csv']
    # names = ['Traffic.csv', 'TrafficTwoMonth.csv']
    # names = get_files(f"{RAW_DIR}")

    for name in names:
        full_df = load_csv(f"{RAW_DIR}/{name}")

    # Date is low-related since we don't know the exact date.
    # So, time and dat of week will be determined as valid feature.

    valid_columns = ['Time', 'Day of the week', 'Total']
                    # 'CarCount', 'BikeCount', 'BusCount', 'TruckCount'
    valid_df = full_df[valid_columns]

    save_csv(f"{CSV_DIR}/full_features.csv", valid_df)


if __name__ == "__main__":
    extract()
