import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'core'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'core/utils'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'core/liblinear_multicore'))

import argparse
import logging
from pathlib import Path

from core.classification import Classification
from core.utils.csv_utils import CsvUtils
from core.utils.core_utils import CoreUtils


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
                       '--delimiter',
                       help = 'csv delimiter, use c for comma and sc for semicolon',
                       type = str,
                       default = 'c')
    parser.add_argument('-t',
                        '--text',
                        help = 'enables documents saving along with the prediction labels inside "predictions.csv" file.',
                        action = "store_true")
    parser.add_argument('-m',
                        '--model',
                        help = 'prediction model (default = Senti4SD.model)',
                        type = str,
                        default = "Senti4SD.model")
    parser.add_argument('-c',
                        '--chunk-size',
                        help = 'chunk size  (--default = 1000)',
                        type = int,
                        default = 1000)
    parser.add_argument('-j',
                        '--jobs-number',
                        help = 'number of jobs for parallelism  (default = 1)',
                        type = int,
                        default = 1)
    parser.add_argument('-o', 
                        '--output',
                        help = 'prediction file name',
                        type = str,
                        default = 'predictions.csv')
    args = parser.parse_args()

    #TODO Add again second input line
    if len(args.input) == 2:
        jar_csv = args.input[0]
        input_csv = args.input[1]
        jar_csv = Path(jar_csv).resolve()
        input_csv = Path(input_csv).resolve()
    elif len(args.input) > 2:
        logging.error("Too many input file. [jar generated csv][input csv]")
        sys.exit(1)
    elif len(args.input) < 2:
        print("Two input file are required. [jar generated csv][input csv]")
        sys.exit(1)
        
    try:
        CsvUtils.check_csv(jar_csv)
        CsvUtils.check_csv(input_csv)
    except OSError as e:
        logging.error(e)
        sys.exit(1)

    if not Path(args.model).exists():
        print("Model doesn't exist. Provide a correct path to the model, or train a new one using the train script.")
        sys.exit(1)

    output_path = Path(f"{Path.cwd()}/predictions")
    output_path.mkdir(parents = True, exist_ok = True )
    output_path = f"{output_path.resolve()}/{args.output}"
    classification = Classification(args.model)
    logging.info("Starting classification task")
    classification.predict(jar_csv, args.chunk_size, CoreUtils.check_jobs_number(args.jobs_number), output_path)
    logging.info("Ending classification task")
    logging.info("Starting ordering prediction csv")
    CsvUtils.order_csv(output_path, 'ID')
    logging.info("Ending ordering prediction csv")
    logging.info("Starting rewriting prediction csv")
    if args.delimiter.lower() == 'c':
        classification.write_id_and_text(input_csv, ',', output_path, args.text)
    elif args.delimiter.lower() == 'sc':
        classification.write_id_and_text(input_csv, ';', output_path, args.text)
    logging.info("Ending rewriting prediction csv")



if __name__ == '__main__':
    main()
    