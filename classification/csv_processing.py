import csv
import argparse
import logging
import sys
from utils import check_csv

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

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
    input_csv = args.input
    input_csv_no_ext = input_csv.split('.')[0]
    try:
        check_csv(input_csv)
        logging.info("Start formatting csv file")
        with open(input_csv, 'r+', newline = '') as csv_file:
            try: 
                csv_delimiter = csv.Sniffer().sniff(csv_file.read(2048)).delimiter
            except csv.Error as e:
                csv_delimiter = ';'
            csv_file.seek(0)
            csv_file_reader =  csv.reader(csv_file, delimiter = csv_delimiter)
            header = next(csv_file_reader)
            if len(header) == 2:
                if header[0].lower() == 'id' and header[1].lower() == 'text':
                    text_rows = [row[1] for row in csv_file_reader]
                    write_csv(input_csv_no_ext, csv_delimiter, text_rows)
                    logging.info("End formatting csv files")
                else:
                    logging.error("Wrong header name in {}".format(input_csv))
            elif len(header) == 1:
                if(header[0].lower() == 'text'):
                    text_rows = [row[0] for row in csv_file_reader]
                    write_csv(input_csv_no_ext, csv_delimiter, text_rows)
                    logging.info("End formatting csv files")
                else:
                    logging.error("Wrong header name in {}".format(input_csv))
            elif len(header) == 0:
                logging.error("{} is empty.".format(input_csv))
                sys.exit(1)
            else:
                logging.error("Too many header in {}".format(input_csv))
                sys.exit(1)
        csv_file.close()
    except OSError as e:
        logging.error(e)
    
if __name__ == '__main__':
    main()