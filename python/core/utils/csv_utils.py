import csv
import os
from multiprocessing import Pool

import pandas as pd
import numpy as np

class CsvUtils():

    @staticmethod
    def __check_file_existence(file_path):
        if not os.path.isfile(file_path):
            return False
        else:
            return True

    @staticmethod       
    def __check_file_extension(file_path, allowed_extension):
        extension = os.path.splitext(file_path)[1]
        if extension not in allowed_extension:
            return False
        else:
            return True

    @staticmethod       
    def check_csv(csv_path):
        if not CsvUtils.__check_file_existence(csv_path):
            raise OSError ("FILE NOT FOUND : {} wasn't found".format(csv_path))
        if not CsvUtils.__check_file_extension(csv_path, ['.csv']):
            raise OSError("WRONG FILE EXTENSION : {} wasn't a csv file.".format(csv_path))

    @staticmethod
    def convert_lines(rows):
        X = np.array([])
        y = np.array([])
        first = True
        for i in range(0, len(rows)):
            values = rows[i].split(',')
            splitted_row_features = [float(value) for value in values[1:-2]]
            splitted_row_label = values[-1].rstrip('\n')
            if first:
                X = np.array(splitted_row_features)
                y = np.array(splitted_row_label)
                first = False
            else:
                X = np.append(X, np.array(splitted_row_features))
                y = np.append(y, np.array(splitted_row_label))
        return X.reshape((i+1, len(splitted_row_features))), y

    @staticmethod
    def from_csv(csv_file, chunk_size, jobs_number):
        stop = False
        rows = []
        chunk_size = int(chunk_size / jobs_number)
        with open(csv_file, 'r+') as csv:
            next(csv)
            while not stop:
                read_rows = []
                try:
                    for _ in range(jobs_number):
                        temp_rows = []
                        for _ in range (chunk_size):
                            temp_rows.append(next(csv))
                        read_rows.append(temp_rows)
                except StopIteration:
                    stop = True
                    read_rows.append(temp_rows)
                finally:
                    if len(temp_rows) != 0:
                        with Pool(jobs_number) as p:
                            results = p.map(CsvUtils.convert_lines, read_rows)
                        for result in results:
                            rows.append(result)
        csv.close()
        first = True
        for row in rows:
            if first:
                X = row[0]
                y = row[1]
                first = False
            else:
                X = np.concatenate((X, row[0]))
                y = np.concatenate((y, row[1]))
        return X, y

    @staticmethod
    def write_to_csv(data, output_csv, csv_delimiter, print_header = False, mode = 'w+'):
        with open(output_csv, mode, newline = '', encoding='utf8') as csv_file:
            csv_file_writer = csv.writer(csv_file, delimiter = csv_delimiter)
            if print_header == True:
                header = data.keys()
                csv_file_writer.writerow(header)
            data = zip(*data.values())
            csv_file_writer.writerows(data)
        csv_file.close()

    @staticmethod
    def order_csv(input_csv, column_name):
        #csv_delimiter = CsvUtils.find_csv_delimiter(input_csv)
        temp = pd.read_csv(input_csv, delimiter = ',')
        temp = temp.sort_values(by=[column_name])
        temp.to_csv(input_csv, index = False)
