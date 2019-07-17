import csv
from collections import OrderedDict

from csv_utils import CsvUtils

class CsvFormatter():

    def __init__(self, header_list, csv_delimiter, header = False):
        self.header_list = header_list
        self.header = header
        self.csv_delimiter = csv_delimiter
    
    def get_rows(self, input_csv):
        with open(input_csv, 'r+', newline = '', encoding='utf8') as csv_file:
            header_list_copy = self.header_list.copy()
            csv_file.seek(0)
            csv_file_reader =  csv.reader(csv_file, delimiter = self.csv_delimiter)
            header = next(csv_file_reader)
            rows = OrderedDict()
            if len(header) == 0:
                csv_file.close()
                raise IOError("{} is empty.".format(input_csv))
            elif len(self.header_list) <= len(header) : 
                count = 0
                while len(header_list_copy) != 0:
                    for i in range(0, len(header)):
                        if header[i].lower().strip() == header_list_copy[0].lower().strip():
                            rows.update({header_list_copy[0]: [row[i] for row in csv_file_reader]})
                            count += 1
                            break
                    header_list_copy.pop(0)
                    csv_file.seek(0)
                    next(csv_file_reader)
                if count != len(self.header_list):
                    csv_file.close()
                    raise IOError("{} not found in {}".format(header_list_copy, input_csv))
            else:
                csv_file.close()
                raise IOError("Too many header in the list.")
        csv_file.close()
        return rows

    def write(self, data, output_csv):
        CsvUtils.write_to_csv(data, output_csv, self.csv_delimiter)
