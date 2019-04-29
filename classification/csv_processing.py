import argparse
import logging
import sys

#from utils import check_csv
from utils.csv_formatter import CsvFormatter
from utils.csv_utils import CsvUtils


logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

def main():
    parser = argparse.ArgumentParser(description = "Csv file processing")
    parser.add_argument('-i',
                       '--input',
                       help = "path to csv file",
                       type = str,
                       required = True)
    args = parser.parse_args()
    input_csv = args.input
    output_csv = "{}_jar.csv".format(input_csv.split('.')[0])
    try:
        CsvUtils.check_csv(input_csv)
        logging.info("Start formatting csv file")
        try:
            csvFormatter = CsvFormatter(['text'])
            data = csvFormatter.get_rows(input_csv)
            csvFormatter.write(data, output_csv)
        except IOError as e:
            logging.error(e)
            sys.exit(1)
        logging.info("End formatting csv file")
    except OSError as e:
        logging.error(e)
        sys.exit(1)
    
if __name__ == '__main__':
    main()
