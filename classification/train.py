import logging
import argparse
import pickle
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import scipy
from liblinearutil import *

from utils.csv_utils import CsvUtils


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
                        help = 'chunk size  --default = 200',
                        type = int,
                        default = 200)
    parser.add_argument('-j', 
                        '--jobs',
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
    
    if len(args.input) == 1:
        try:
            logging.info("Start reading dataset in chunk...")
            X, y = CsvUtils.from_csv(args.input[0], args.chunk_size)
            logging.info("End reading dataset in chunk...")
        except OSError as e:
            print(e)
            sys.exit(1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, stratify = y, random_state = seed)
        del X, y
    elif len(args.input) == 2:
        
        #Check file existence in advance to avoid missing test set
        try:
            CsvUtils.check_csv(args.input[0])
            CsvUtils.check_csv(args.input[1])
        except OSError as e:
            print(e)
            sys.exit(1)

        #read the train set in chunk
        logging.info("Start reading training set in chunk...")
        X_train, y_train = CsvUtils.from_csv(args.input[0], args.chunk_size)
        logging.info("End reading training set in chunk...")
    
    else:
        print("Too many input arguments.")

    logging.info("Start encoding training set labels..")
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    logging.info("End econding training set labels..")


    prob = problem(y_train, scipy.sparse.csr_matrix(X_train))

    S_VALUE = [0, 1, 2, 3, 4, 5, 6, 7]
    C_VALUE = [0.01, 0.05, 0.10, 0.20, 0.25, 0.50, 1, 2, 4, 8]

    best_s_value = 0
    best_c_value = 0
    best_cv_accuracy = 0

    current_c_value = 0
    current_cv_accuracy = 0

    #search best parameters
    logging.info("Start parameter tuning...")
    for s_value in S_VALUE:
        for c_value in C_VALUE:
            parameters = "-s {} -c {} -v 10 -B 1 -e 0.00001 -q".format(s_value, c_value)
            param = parameter(parameters)
            cv_accuracy = train(prob, param)
            if cv_accuracy > current_cv_accuracy:
                current_cv_accuracy = cv_accuracy
                current_c_value = c_value
            if cv_accuracy > best_cv_accuracy:
                best_s_value = s_value
                best_c_value = c_value
                best_cv_accuracy = cv_accuracy
        print(f"Solver {s_value}")
        print(f"Best accuracy {current_cv_accuracy}")
        print(f"Best C {current_c_value}")
        current_c_value = 0
        current_cv_accuracy = 0
    logging.info("End parameter tuning...")
        

    #retrain the model
    parameters = "-s {} -c {} -B 1".format(best_s_value, best_c_value)
    param = parameter(parameters)
    m = train(prob, param)

    print(best_s_value)
    print(best_c_value)
    print(best_cv_accuracy)
        

    del X_train, y_train
        
    if len(args.input) == 2:   
        logging.info("Start reading test set in chunk...")
        X_test, y_test = CsvUtils.from_csv(args.input[1], args.chunk_size)
        logging.info("End reading test set in chunk...")

    logging.info("Start encoding training set labels..")
    y_test = le.transform(y_test)
    logging.info("End econding training set labels..")
        
    p_label, p_acc, p_val = predict(y_test, X_test, m)

    print(p_label[0])
    p_label = [int(label) for label in p_label]
    print(p_label[0])

    y_test = le.inverse_transform(y_test)
    p_label = le.inverse_transform(p_label)
    np.array(p_label)

    logging.info(classification_report(y_test, p_label))

    save_model('./{}.model'.format(args.model), m)
    filehandler = open('./{}.label'.format("Senti4SD_label"), 'wb')
    pickle.dump(le, filehandler)
    
   
if __name__ == '__main__':
    main()
                        
