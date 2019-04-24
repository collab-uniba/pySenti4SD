import csv
import argparse
from utils import check_csv

def write_csv(output_file, csv_delimiter, text_rows):
    with open("{}_jar.csv".format(output_file), 'w+', newline = '') as csv_file:
        csv_file_writer = csv.writer(csv_file, delimiter = csv_delimiter)
        for text in text_rows:
            csv_file_writer.writerow([text])
    csv_file.close()

def main():
    parser = argparse.ArgumentParser(description = "Csv file processing")
    parser.add_argument('-i',
                       '--input',
                       help = "path to csv file",
                       type = str,
                       required = True)
    args = parser.parse_args()
    input_file = args.input
    input_file_no_ext = input_file.split('.')[0]
    try:
        check_csv(input_file)
        with open(input_file, 'r+', newline = '') as csv_file:
            try: 
                csv_delimiter = csv.Sniffer().sniff(csv_file.read(2048)).delimiter
                print("delimiter = '{}'".format(csv_delimiter))
            except csv.Error as e:
                print("Can't find delimiter ';' used.")
                csv_delimiter = ';'
            csv_file.seek(0)
            csv_file_reader =  csv.reader(csv_file, delimiter = csv_delimiter)
            header = next(csv_file_reader)
            if len(header) == 2:
                if header[0].lower() == 'id' and header[1].lower() == 'text':
                    text_rows = [row[1] for row in csv_file_reader]
                    write_csv(input_file_no_ext, csv_delimiter, text_rows)
                else:
                    print("Wrong header name for csv file") 
            elif len(header) == 1:
                if(header[0].lower() == 'text'):
                    text_rows = [row[0] for row in csv_file_reader]
                    write_csv(input_file_no_ext, csv_delimiter, text_rows)
            elif len(header) == 0:
                print("Empty csv file.")
            else:
                print("Too many header inside csv file")
        csv_file.close()
    except OSError as e:
        print(e)
    
if __name__ == '__main__':
    main()