import os
import csv
import glob
from multiprocessing import Pool
from collections import OrderedDict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder

from utils.csv_utils import CsvUtils
from utils.csv_formatter import CsvFormatter

from liblinearutil import *

class Classification():

    def __init__(self, model):
        self.model = model

    def __create_classification_file(self, pred_csv):
        with open(pred_csv, 'w+') as prediction:
            prediction.write("ID,PREDICTED\n")
        prediction.close()

    def __clean_id(self, id):
        temp = id.split(',')[0]
        temp = temp.replace('t', "")
        return int(temp)

    def __convert_lines_and_predict(self, rows, label_encoder, pred_file):
        model = load_model(self.model)
        X = np.array([])
        splitted_rows_id = []
        first = True
        for i in range(0, len(rows)):
            values = rows[i].split(',')
            splitted_rows_id.append(values[0])
            splitted_row_features = [float(value) for value in values[1:]]
            if first:
                X = np.array(splitted_row_features)
                first = False
            else:
                X = np.append(X, np.array(splitted_row_features))
        X = X.reshape((i+1, len(splitted_row_features)))
        y_pred, y_acc, y_val = predict([], X, model, '-q')
        y_pred = [int(label) for label in y_pred]
        y_pred = label_encoder.inverse_transform(y_pred)
        y_pred = [pred.replace('\n', "") for pred in y_pred]
        dataframe = OrderedDict()
        dataframe.update({'id': [(self.__clean_id(row_id) + 1) for row_id in splitted_rows_id]})
        dataframe.update({'predicted' : y_pred})
        CsvUtils.write_to_csv(dataframe, pred_file, ',', False, 'a+')
    
    def predict(self, csv_file, chunk_size, jobs_number, pred_file):
        self.__create_classification_file(pred_file)
        chunk_size = int(chunk_size / jobs_number)
        stop = False
        label_encoder = LabelEncoder()
        label_encoder.fit(['positive', 'negative', 'neutral'])
        with open(csv_file, 'r+') as csv:
            next(csv)
            while not stop:
                read_rows = []
                try:
                    for _ in range(jobs_number):
                        temp_rows = []
                        for _ in range (chunk_size):
                            temp_rows.append(next(csv))
                        read_rows.append(temp_rows)
                except StopIteration:
                    stop = True
                    read_rows.append(temp_rows)
                finally:
                    Parallel(n_jobs = jobs_number)(delayed(self.__convert_lines_and_predict)(rows, label_encoder, pred_file) for rows in read_rows)
        csv.close()

    def write_id_and_text(self, input_csv, csv_delimiter, pred_csv, text = False):
        dataframe = OrderedDict()
        try:
            csv_fomatter = CsvFormatter(['ID'], csv_delimiter)
            dataframe.update(csv_fomatter.get_rows(input_csv))
        except IOError as e:
            print(e)
        if text:
            try:
                csv_fomatter = CsvFormatter(['TEXT'], csv_delimiter)
                dataframe.update(csv_fomatter.get_rows(input_csv))
            except IOError as e:
                print(e)
        if dataframe:
            temp = pd.read_csv(pred_csv, delimiter = ",")
            dataframe.update({'PREDICTED': temp.iloc[:, -1:].values.ravel()})
            CsvUtils.write_to_csv(dataframe, pred_csv, ',', True)
