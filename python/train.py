import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'core'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'core/utils'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'core/liblinear_multicore'))

import logging
import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from core.tuning_parameter import Tuning
from core.train_model import Train
from core.utils.csv_utils import CsvUtils
from core.utils.core_utils import CoreUtils


logging.basicConfig(level = logging.INFO, format = "[%(levelname)s] %(asctime)s - %(message)s")
    

def main():
    parser = argparse.ArgumentParser(description = "Hyperparameter tuning")
    parser.add_argument('-i',
                       '--input',
                       help = "path to train set and to test set csv.",
                       type = str,
                       action = 'append',
                       required = True)
    parser.add_argument('-c',
                        '--chunk-size',
                        help = 'chunk size  --default = 1000',
                        type = int,
                        default = 1000)
    parser.add_argument('-j', 
                        '--jobs-number',
                        help = 'number of jobs',
                        type = int,
                        default = 1)
    parser.add_argument('-m', 
                        '--model',
                        help = 'model file name',
                        type = str,
                        default = 'Senti4SD')
    args = parser.parse_args()

    seed = np.random.seed(42)

    jobs_number = CoreUtils.check_jobs_number(args.jobs_number)
    
    if len(args.input) == 1:

        train_file_path = Path(args.input[0]).resolve()

        # Check file existence in advance to avoid missing test set
        try:
            CsvUtils.check_csv(train_file_path)
        except OSError as e:
            print(e)
            sys.exit(1)

        try:
            logging.info("Start reading dataset in chunk...")
            X, y = CsvUtils.from_csv(train_file_path, args.chunk_size, jobs_number)
            logging.info("End reading dataset in chunk...")
        except OSError as e:
            print(e)
            sys.exit(1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, stratify = y, random_state = seed)
        del X, y
    elif len(args.input) == 2:

        train_file_path = Path(args.input[0]).resolve()
        test_file_path = Path(args.input[1]).resolve()
        
        #Check file existence in advance to avoid missing test set
        try:
            CsvUtils.check_csv(train_file_path)
            CsvUtils.check_csv(test_file_path)
        except OSError as e:
            print(e)
            sys.exit(1)

        #read the train set in chunk
        logging.info("Start reading training set in chunk...")
        X_train, y_train = CsvUtils.from_csv(train_file_path, args.chunk_size, jobs_number)
        logging.info("End reading training set in chunk...")
        logging.info("Start reading test set in chunk...")
        X_test, y_test = CsvUtils.from_csv(test_file_path, args.chunk_size, jobs_number)
        logging.info("End reading test set in chunk...")
    
    else:
        print("Too many input arguments.")

    #create path
    output_path = Path('liblinear_perfomance')
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path.resolve()
    dir_path = output_path.parent
    model_path = f"{dir_path}/{args.model}.model"

    logging.info("Start parameter tuning")
    current_path = Path.cwd()
    solvers_path = Path(f'{current_path}/liblinear_solvers').resolve()
    tuning = Tuning(jobs_number, solvers_path, output_path)
    best_solver_name, best_solver_value, best_c_value = tuning.tuning_parameter(X_train, X_test, y_train, y_test)
    logging.info("End parameter tuning")

    logging.info("Start training model")
    train = Train(jobs_number, best_solver_name, best_solver_value, best_c_value, model_path)
    train.train_model(X_train, X_test, y_train, y_test)
    train.save_best_perfomance(dir_path)
    logging.info("End training model")
    
   
if __name__ == '__main__':
    main()
                        
