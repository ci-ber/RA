import csv
import numpy as np
import tqdm
import glob


def generate_csv(img_path, output_path, dataset_name):
    """
    Generate Splits and csv from a folder with images
    @param: img_path: str
        path to images
    @param: output_path: str
        path to output train csv
    """
    train_path = output_path + dataset_name + '_train.csv'
    val_path = output_path + dataset_name + '_val.csv'
    test_path = output_path + dataset_name + '_test.csv'

    np.random.seed(2109)
    train_keys = glob.glob(img_path)
    ratio_test = int(0.1 * len(train_keys))  # 10% val; 10% test
    val_keys = np.random.choice(train_keys, 2 * ratio_test, replace=False)
    test_keys = np.random.choice(val_keys, ratio_test, replace=False)
    train_files, val_files, test_files = [], [], []
    for scan in train_keys:
        if scan in test_keys:
            test_files.append([scan])
        elif scan in val_keys:
            val_files.append([scan])
        else:
            train_files.append([scan])
    top_row = ['filename']
    write_csv(train_files, train_path, top_row)
    write_csv(val_files, val_path, top_row)
    write_csv(test_files, test_path, top_row)


def write_csv(file, path, top_row):
    """
    Write files to csv
    """
    with open(path, 'w') as csvfile:
        csvW = csv.writer(csvfile, delimiter=',',  quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvW.writerow(top_row)
        for datar in tqdm.tqdm(file):
            csvW.writerow(datar)


def get_data_from_csv(path_to_csv):
    """
    :param path_to_csv: str
        path to csv with filenames
    :return: list
        list with all the filenames
    """
    files = []
    if type(path_to_csv) is not list:
        path_to_csv = [path_to_csv]
    for single_csv in path_to_csv:
        ct = 0
        with open(single_csv, newline='') as csv_file:
            p_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
            for row in p_reader:
                if type(row) == list:
                    row = row[0]
                ct += 1
                if ct == 1:
                    continue
                files.append(row)
    return files
