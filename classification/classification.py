import os
import csv
import glob

import pandas as pd
from joblib import Parallel, delayed

from utils.csv_utils import CsvUtils
from utils.csv_formatter import CsvFormatter

class Classification():

    def __init__(self, model):
        self.model = model

    def __create_classification_file(self, pred_csv):
        with open(pred_csv, 'w+') as prediction:
            prediction.write("ID,PREDICTED\n")
        prediction.close()

    def __clean_id(self, id):
        temp = id.split(",")[0]
        temp = temp.replace("t", "")
        return int(temp)

    def __write_chunk(self, chunk_name, header, lines):
        with open(chunk_name, 'w+') as jar_csv_chunk:
            jar_csv_chunk.write(header)
            jar_csv_chunk.writelines(line for line in lines)
        jar_csv_chunk.close()

    def __classification(self, jar_csv_chunk, pred_file):
        chunk = pd.read_csv(jar_csv_chunk, delimiter = ',')
        chunk.dropna(how="any")
        pred = self.model.predict(chunk.iloc[:, 1:].values)
        row_number = [self.__clean_id(id) for id in chunk['id'].values]
        row_number = [id+1 for id in row_number]
        pred_df = pd.DataFrame({'id': row_number, 'predicted': pred})
        pred_df.to_csv(pred_file, sep = ',', index = False, header = False, mode = 'a')
    

    def create_split_and_predict(self, jar_csv, model, chunk_size, number_of_split, pred_file):
        if not os.path.exists('temp_split'):
            os.mkdir('temp_split')
        self.__create_classification_file(pred_file)
        lines = []
        with open(jar_csv, 'r') as csv_file:
            stop = False
            start_chunk = 0
            stop_chunk = chunk_size
            count = 0
            try:
                header = next(csv_file)
                while not stop:
                    while count < number_of_split:
                        chunk_name = './temp_split/split-{}.csv'.format(count)
                        try: 
                            while start_chunk < stop_chunk:
                                lines.append(next(csv_file))
                                start_chunk += 1
                            self.__write_chunk(chunk_name, header, lines)
                        except StopIteration:
                            stop = True
                            self.__write_chunk(chunk_name, header, lines)
                            csv_file.close()
                            count = number_of_split
                        else:
                            lines = []
                            start_chunk = stop_chunk
                            stop_chunk = stop_chunk + chunk_size
                            count += 1
                    count = 0
                    csv_files = glob.glob('./temp_split/*.csv')
                    Parallel(n_jobs = number_of_split, verbose = 1)(delayed(self.__classification) (csv, pred_file) for csv in csv_files)
                    for csv in csv_files:
                        os.remove(csv)
                os.rmdir('temp_split')
            except StopIteration:
                print('Empty file.')

    def write_id_and_text(self, input_csv, pred_csv, text = False):
        dataframe = {}
        try:
            csv_fomatter = CsvFormatter(['id'])
            dataframe.update(csv_fomatter.get_rows(input_csv))
        except IOError as e:
            print(e)
        if text:
            try:
                csv_fomatter = CsvFormatter(['text'])
                dataframe.update(csv_fomatter.get_rows(input_csv))
            except IOError as e:
                print(e)
        if dataframe:
            temp = pd.read_csv(pred_csv, delimiter = ",")
            dataframe.update({'PREDICTED': temp.iloc[:, -1:].values.ravel()})
            CsvUtils.write_to_csv(dataframe, pred_csv, ',', True)
