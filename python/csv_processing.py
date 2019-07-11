import argparse
import logging
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'core/utils'))

from core.utils.csv_formatter import CsvFormatter
from core.utils.csv_utils import CsvUtils

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

def main():
    parser = argparse.ArgumentParser(description = "Csv file processing")
    parser.add_argument('-i',
                       '--input',
                       help = "path to csv file",
                       type = str,
                       required = True)
    parser.add_argument('-d',
                       '--delimiter',
                       help = 'csv delimiter, use c for comma and sc for semicolon',
                       type = str,
                       default = 'c')
    parser.add_argument('-c',
                       '--columns',
                       help = "column or columns to extract from csv [default = 'text']",
                       type = str,
                       action = 'append',
                       required = True)
    args = parser.parse_args()
    input_csv = args.input
    input_csv = Path(input_csv).resolve()
    output_csv = "{}/{}_jar.csv".format(input_csv.parent, input_csv.name.split('.')[0])
    try:
        CsvUtils.check_csv(input_csv)
        logging.info("Start formatting csv file")
        try:
            if(args.delimiter == 'c'):
                csvFormatter = CsvFormatter(args.columns, ',')
            elif(args.delimiter == 'sc'):
                csvFormatter = CsvFormatter(args.columns, ';')
            else:
                logging.error('Wrong csv delimiter. Use "c" for comma and "sc" for semicolon.')
                sys.exit(1)
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
