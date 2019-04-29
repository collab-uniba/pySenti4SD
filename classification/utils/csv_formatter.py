import csv

from utils.csv_utils import CsvUtils 

class CsvFormatter():

    def __init__(self, header_list, header = False):
        self.header_list = header_list
        self.header = header
    
    def get_rows(self, input_csv):
        with open(input_csv, 'r+', newline = '') as csv_file:
            self.csv_delimiter = CsvUtils.find_csv_delimiter(input_csv)
            csv_file.seek(0)
            csv_file_reader =  csv.reader(csv_file, delimiter = self.csv_delimiter)
            header = next(csv_file_reader)
            rows = {}
            if len(header) == 0:
                csv_file.close()
                raise IOError("{} is empty.".format(input_csv))
            elif len(header) >= len(self.header_list): 
                count = 0
                for i in range(0, len(header)):
                    for value in self.header_list:   
                        if header[i].lower().strip() == value.lower().strip():
                            rows.update({value: [row[i] for row in csv_file_reader]})
                            count += 1
                if count != len(self.header_list):
                    csv_file.close()
                    raise IOError("Headers not found in {}".format(input_csv))
            else:
                csv_file.close()
                raise IOError("Too many header in the list.")
        csv_file.close()
        return rows

    def write(self, data, output_csv):
        CsvUtils.write_to_csv(data, output_csv, self.csv_delimiter)
