import csv
import os

import pandas as pd

class CsvUtils(object):

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
    def find_csv_delimiter(input_csv):
        csv_delimiter = ';'
        with open(input_csv, 'r+', newline = '') as csv_file:
            try: 
                csv_delimiter = csv.Sniffer().sniff(csv_file.read(2048)).delimiter
            except csv.Error:
                csv_delimiter = ';'
        csv_file.close()
        return csv_delimiter

    @staticmethod
    def write_to_csv(data, output_csv, csv_delimiter, print_header = False):
        with open(output_csv, 'w+', newline = '') as csv_file:
            csv_file_writer = csv.writer(csv_file, delimiter = csv_delimiter)
            if print_header == True:
                header = data.keys()
                csv_file_writer.writerow(header)
            data = zip(*data.values())
            csv_file_writer.writerows(data)
        csv_file.close()

    @staticmethod
    def order_csv(input_csv, column_name):
        csv_delimiter = CsvUtils.find_csv_delimiter(input_csv)
        temp = pd.read_csv(input_csv, delimiter = csv_delimiter)
        temp = temp.sort_values(by=[column_name])
        temp.to_csv(input_csv, index = False)