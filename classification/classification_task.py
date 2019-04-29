import sys
import argparse
import pickle
import logging
import csv

from classification import Classification
from utils.csv_utils import CsvUtils

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level = logging.INFO)

def main():
    parser = argparse.ArgumentParser(description = "Classification task")
    parser.add_argument('-i',
                       '--input',
                       help = "path to csv file.",
                       type = str,
                       action = 'append',
                       required = True)
    parser.add_argument('-d',
                        '--documents',
                        help = 'write starting documents in prediction csv file (defualt = no)',
                        type = str,
                        default = "no")
    parser.add_argument('-m',
                        '--trained-model',
                        help = 'trained model (default = Senti4SD_model)',
                        type = str,
                        default = "Senti4SD_model.clf")
    parser.add_argument('-c',
                        '--chunk-size',
                        help = 'chunk size  (--default = 1000)',
                        type = int,
                        default = 1000)
    parser.add_argument('-j',
                        '--number-of-jobs',
                        help = 'number of jobs for parallelism  (default = 1)',
                        type = int,
                        default = 1)
    parser.add_argument('-o', 
                        '--output',
                        help = 'prediction file name',
                        type = str,
                        default = 'predictions.csv')
    args = parser.parse_args()

    #TODO check missing file
    filehandler = open(args.trained_model, "rb")
    model = pickle.load(filehandler)

    #TODO Add again second input line

    if len(args.input) == 2:
        jar_csv = args.input[0]
        input_csv = args.input[1]
    elif len(args.input) > 2:
        logging.error("Too many input file. [jar generated csv][input csv]")
        sys.exit(1)
    elif len(args.input) < 2:
        print("Two input file are required. [jar generated csv][input csv]")
        sys.exit(1)

    try:
        CsvUtils.check_csv(jar_csv)
        CsvUtils.check_csv(input_csv)
        classification = Classification(model)
        with open(input_csv, 'r+', newline = '') as csv_file:
            text = True if args.documents == "yes" else False
            logging.info("Starting classification task")
            classification.create_split_and_predict(jar_csv, model, args.chunk_size, args.number_of_jobs, args.output)
            logging.info("Ending classification task")
            logging.info("Starting ordering prediction csv")
            CsvUtils.order_csv(args.output, 'ID')
            logging.info("Ending ordering prediction csv")
            logging.info("Starting rewriting prediction csv")
            classification.write_id_and_text(input_csv, args.output, text)
            logging.info("Ending rewriting prediction csv")
        csv_file.close()
    except OSError as e:
        logging.error(e)
        sys.exit(1)


if __name__ == '__main__':
    main()