import os
import pickle
import logging
import csv
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.externals.joblib import Memory

logging.basicConfig(level = logging.INFO, format = "[%(levelname)s] %(asctime)s - %(message)s")
        
        
def check_file_existence(file_path):
    if not os.path.isfile(file_path):
        return False
    else:
        return True
        
def check_file_extension(file_path, allowed_extension):
    extension = os.path.splitext(file_path)[1]
    if extension not in allowed_extension:
        return False
    else:
        return True
        
def check_csv(csv_path):
    if not check_file_existence(csv_path):
        raise OSError ("FILE NOT FOUND : {} wasn't found".format(csv_path))
    if not check_file_extension(csv_path, ['.csv']):
        raise OSError("WRONG FILE EXTENSION : {} wasn't a csv file.".format(csv_path))
            
def from_csv(csv_path, chunk_size, verbose = 0):
    if verbose == 1:
        logging.info("Start reading training set in chunks...")
        start = datetime.now()
    X = np.array([])
    y = np.array([])
    i = 0
    for chunk in pd.read_csv(csv_path, chunksize = chunk_size, delimiter = ','):
        chunk = chunk.dropna(how = "any")
        if len(X) == 0 and len(y) == 0:
            X = chunk.iloc[:, 1:-1].values
            y = chunk.iloc[:, -1:].values.ravel()
        else:
            X = np.concatenate((X, chunk.iloc[:, 1:-1].values))
            y = np.concatenate((y, chunk.iloc[:, -1:].values.ravel()))
        if verbose == 1:
            logging.info("Chunk {} readed".format(i))
        i += 1
    del chunk
    if verbose == 1:
        end = datetime.now() - start
        logging.info("Elapsed time : {}".format(end))
    return X, y 

def save_classifier(clf, file_name = "Senti4SD_model"):
    #TODO handle missing folder exception
    filehandler = open('{}.clf'.format(file_name), 'wb')
    pickle.dump(clf, filehandler)
    logging.info("Model saved as {}".format(file_name))
    
def load_classifier(file_name = "Senti4SD_model"):
    #TODO handle missing file exception
    filehandler = open('../{}.clf'.format(file_name), 'rb')
    return pickle.load(filehandler)
    
def get_id(line):
    temp = line.split(",")[0]
    temp = temp.replace("t", "")
    return int(temp)

def save_params(model):
    with open('best_params.txt', 'w+') as best_params:
        for k, v in model.get_params().items():
            best_params.write("{} : {} \n".format(str(k), str(v)))
    best_params.close()