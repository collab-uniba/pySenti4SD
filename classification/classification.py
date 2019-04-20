import os
import glob
import argparse
import pickle
import csv
from datetime import datetime

import pandas as pd
from sklearn.externals.joblib import Parallel,delayed

from utils import load_classifier, check_csv, get_id, find_lines


def create_prediction_file(dictionary, pred_file):
    with open(pred_file, 'w+') as prediction:
        if dictionary is None:
            prediction.write("ROW,PREDICTED \n")
        else:
            prediction.write("ROW,DOCUMENT,PREDICTED \n")
    prediction.close()

def classification(model, csv_file, dictionary, pred_file):
    chunk = pd.read_csv(csv_file, delimiter = ',')
    chunk.dropna(how="any")
    pred = model.predict(chunk.iloc[:, 1:].values)
    lines_number = [get_id(id) for id in chunk['id'].values]
    print(lines_number)
    if dictionary is not None: 
        lines = find_lines(dictionary, lines_number)
        lines_number = [id+1 for id in lines_number]
        pred_df = pd.DataFrame({'id': lines_number, 'document': lines, 'predicted': pred})
    else:
        lines_number = [id+1 for id in lines_number]
        pred_df = pd.DataFrame({'id': lines_number, 'predicted': pred})
    pred_df.to_csv(pred_file, sep = ',', index = False, header = False, mode = 'a', quoting=csv.QUOTE_ALL)
    

def create_split_and_predict(file, dictionary, model, chunk_size, number_of_split, pred_file):
    #TODO Caricare il classificatore corretto
    print("Start classification...")
    start = datetime.now()
    if not os.path.exists('temp_split'):
        os.mkdir('temp_split')
    temp = []
    with open(file, 'r') as csv_file:
        stop = False
        start_chunk = 0
        stop_chunk = chunk_size
        count = 0
        try:
            create_prediction_file(dictionary, pred_file)
            header = next(csv_file)
            while not stop:
                while count < number_of_split:
                    try: 
                        while start_chunk < stop_chunk:
                            temp.append(next(csv_file))
                            start_chunk += 1
                        chunk_file = './temp_split/split-{}.csv'.format(count)
                        with open(chunk_file, 'w+') as chunk_file:
                            chunk_file.write(header)
                            chunk_file.writelines(line for line in temp)
                        chunk_file.close()
                        temp = []
                        start_chunk = stop_chunk
                        stop_chunk = stop_chunk + chunk_size
                        count += 1
                    except StopIteration:
                        stop = True
                        chunk_file = './temp_split/split-{}.csv'.format(count)
                        with open(chunk_file, 'w+') as chunk_file:
                            chunk_file.write(header)
                            chunk_file.writelines(line for line in temp)
                        chunk_file.close()
                        csv_file.close()
                        count = number_of_split
                count = 0
                csv_files = glob.glob('./temp_split/*.csv')
                Parallel(n_jobs = number_of_split, verbose = 1)(delayed(classification) (model, csv, dictionary, pred_file) for csv in csv_files)
                for csv in csv_files:
                    os.remove(csv)
            end = datetime.now() - start
            print("Elapsed time : {}".format(end))
            os.rmdir('temp_split')
        except StopIteration:
            print('Empty file.')

def order_csv(pred_file):
    temp = pd.read_csv(pred_file, delimiter = ",")
    temp = temp.sort_values(by=['ROW'])
    temp.to_csv(pred_file, index = False)
            
def main():
    parser = argparse.ArgumentParser(description = "Classification")
    parser.add_argument('-i',
                       '--input',
                       help = "path to train set and to test set csv.",
                       type = str,
                       action = 'append',
                       required = True)
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
                        default = 'prediction.csv')
    args = parser.parse_args()

    #TODO check missing file
    filehandler = open(args.trained_model, "rb")
    model = pickle.load(filehandler)

    if len(args.input) == 1:
        try:
            check_csv(args.input[0])
            create_split_and_predict(args.input[0], None, model, args.chunk_size, args.number_of_jobs, args.output)
            order_csv(args.output)
        except OSError as e:
            print(e)
    elif len(args.input) == 2:
        
        try:
            check_csv(args.input[0])
            check_csv(args.input[1])
            create_split_and_predict(args.input[0], args.input[1], model, args.chunk_size, args.number_of_jobs, args.output)
            order_csv(args.output)
        except OSError as e:
            print(e)
    
    else:
        print("Too many input arguments.")

if __name__ == '__main__':
    main()
