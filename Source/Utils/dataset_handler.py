import os
import pandas as pd
from datasets import __file__ as datasets_init_path

DATASETS_FOLDER_PATH = os.path.dirname(datasets_init_path)
FIRST_DATASET_PATH = os.path.join(DATASETS_FOLDER_PATH, '3D_spatial_network.txt')
SECOND_DATASET_PATH = os.path.join(DATASETS_FOLDER_PATH, 'household_power_consumption.txt')
THIRD_DATASET_PATH = os.path.join(DATASETS_FOLDER_PATH, 'kc_house_data.csv')

DATASET_LIST = [FIRST_DATASET_PATH, SECOND_DATASET_PATH, THIRD_DATASET_PATH]


def get_dataset(dataset_path):
    if dataset_path == FIRST_DATASET_PATH:
        return _read_first_dataset()
    if dataset_path == SECOND_DATASET_PATH:
        return _read_second_dataset()
    if dataset_path == THIRD_DATASET_PATH:
        return _read_third_dataset()
    raise ValueError("Unknown dataset: {}".format(dataset_path))


def _read_first_dataset():
    df = pd.read_csv(FIRST_DATASET_PATH, names=['OSM_ID', 'LONGITUDE', 'LATITUDE', 'ALTITUDE'])
    A_tag = df.iloc[:, 1:4].as_matrix()
    return A_tag


def _read_second_dataset():
    df = pd.read_csv(SECOND_DATASET_PATH, sep=';')
    df = df[~df.eq('?').any(1)]  # There are 25,979 missing entries the authors forgot to mention.
    return df.iloc[:, 2:5].astype(float).as_matrix()


def _read_third_dataset():
    # In the third dataset I've found 21613 values instead of 21600 values. Will take the first 21600 values
    # to be as close to the article.
    df = pd.read_csv(THIRD_DATASET_PATH)
    #  3 - Bedrooms
    #  5 - sqft living
    #  6 - sqft lot
    #  7 - floors
    #  8 - waterfront
    # 12 - sqft above
    # 13 - sqft basement
    # 14 - year_built
    # 2 - price (b vector, what we wish to predict)
    return df.iloc[:21600, [3, 5, 6, 7, 8, 12, 13, 14, 2]].as_matrix()
