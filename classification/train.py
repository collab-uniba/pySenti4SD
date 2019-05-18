import logging
import argparse
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import scipy
from liblinearutil import *

from utils.csv_utils import CsvUtils
from utils.core_utils import CoreUtils


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

    #search best parameters
    logging.info("Start parameter tuning...")
    jobs_number = CoreUtils.check_jobs_number(args.jobs_number)

    solvers = {
        0: "L2-regularized logistic regression (primal)",
		1: "L2-regularized L2-loss support vector classification (dual)",
		2: "L2-regularized L2-loss support vector classification (primal)",
		3: "L2-regularized L1-loss support vector classification (dual)",
		4: "support vector classification by Crammer and Singer",
		5: "L1-regularized L2-loss support vector classification",
		6: "L1-regularized logistic regression",
	    7: "L2-regularized logistic regression (dual)"
    }

    C_VALUE = [0.01, 0.05, 0.10, 0.20, 0.25, 0.50, 1, 2, 4, 8]

    cv_accuracy = 0
    best_cv_accuracy = 0
    best_c_value = 0
    best_s_value = 0

    prob = problem(y_train, X_train)
    for solver_value, solver_name in solvers.items():
        print(f"Tuning solver {solver_name}")
        for c_value in C_VALUE:
            print(f"C value: {c_value}")
            if solver_value == 4 or solver_value == 7:
                parameters = "-s {} -c {} -v 10 -B 1 -q".format(solver_value, c_value)
            else:
                parameters = "-s {} -n {} -c {} -v 10 -B 1 -q".format(solver_value, jobs_number, c_value)
            param = parameter(parameters)
            cv_accuracy = train(prob, param)
            if cv_accuracy > best_cv_accuracy:
                best_c_value = c_value
                best_cv_accuracy = cv_accuracy
                best_s_value = solver_value 
    #solver_results = Parallel(n_jobs = jobs_number)(delayed(search_best_parameter)(X_train, y_train, s_value, C_VALUE) for s_value in S_VALUE)
    logging.info("End parameter tuning...")
        
    logging.info(f"Best solver: {solvers[best_s_value]}")
    logging.info(f"Best C value: {best_c_value}")
    logging.info(f"Best cv accuracy: {best_cv_accuracy}")

    logging.info("Train model with selected solver and best C value...")
    #retrain the model
    if best_s_value == 4 or best_s_value == 7:
        parameters = "-s {} -c {} -B 1".format(best_s_value, best_c_value)
    else:
        parameters = "-s {} -n {} -c {} -B 1 -q".format(best_s_value, jobs_number, best_c_value)
    param = parameter(parameters)
    m = train(prob, param)
    logging.info("End retraining model")
        

    del X_train, y_train
        
    if len(args.input) == 2:   
        logging.info("Start reading test set in chunk...")
        X_test, y_test = CsvUtils.from_csv(args.input[1], args.chunk_size)
        logging.info("End reading test set in chunk...")

    logging.info("Start encoding training set labels..")
    y_test = le.transform(y_test)
    logging.info("End econding training set labels..")
        
    p_label, p_acc, p_val = predict(y_test, X_test, m)

    #Convert predicted value from float to int
    p_label = [int(label) for label in p_label]

    y_test = le.inverse_transform(y_test)
    p_label = le.inverse_transform(p_label)
    np.array(p_label)

    logging.info("Classification report on test set")
    print(classification_report(y_test, p_label))

    save_model('./{}.model'.format(args.model), m)
    
   
if __name__ == '__main__':
    main()
                        
