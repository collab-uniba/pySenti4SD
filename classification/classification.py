import os
import glob
import argparse
import pickle
import csv
import sys
from datetime import datetime

import pandas as pd
from sklearn.externals.joblib import Parallel,delayed

from utils import load_classifier, check_csv, get_id


def create_prediction_file(pred_csv):
    with open(pred_csv, 'w+') as prediction:
        prediction.write("ID,PREDICTED\n")
    prediction.close()

def classification(model, jar_csv_chunk, pred_file):
    chunk = pd.read_csv(jar_csv_chunk, delimiter = ',')
    chunk.dropna(how="any")
    pred = model.predict(chunk.iloc[:, 1:].values)
    lines_number = [get_id(id) for id in chunk['id'].values]
    lines_number = [id+1 for id in lines_number]
    pred_df = pd.DataFrame({'id': lines_number, 'predicted': pred})
    pred_df.to_csv(pred_file, sep = ',', index = False, header = False, mode = 'a', quoting=csv.QUOTE_ALL)
    

def create_split_and_predict(jar_csv, model, chunk_size, number_of_split, pred_file):
    #TODO Caricare il classificatore corretto
    print("Start classification...")
    start = datetime.now()
    if not os.path.exists('temp_split'):
        os.mkdir('temp_split')
    temp = []
    with open(jar_csv, 'r') as csv_file:
        stop = False
        start_chunk = 0
        stop_chunk = chunk_size
        count = 0
        try:
            header = next(csv_file)
            while not stop:
                while count < number_of_split:
                    try: 
                        while start_chunk < stop_chunk:
                            temp.append(next(csv_file))
                            start_chunk += 1
                        jar_csv_chunk = './temp_split/split-{}.csv'.format(count)
                        with open(jar_csv_chunk, 'w+') as jar_csv_chunk:
                            jar_csv_chunk.write(header)
                            jar_csv_chunk.writelines(line for line in temp)
                        jar_csv_chunk.close()
                        temp = []
                        start_chunk = stop_chunk
                        stop_chunk = stop_chunk + chunk_size
                        count += 1
                    except StopIteration:
                        stop = True
                        jar_csv_chunk = './temp_split/split-{}.csv'.format(count)
                        with open(jar_csv_chunk, 'w+') as jar_csv_chunk:
                            jar_csv_chunk.write(header)
                            jar_csv_chunk.writelines(line for line in temp)
                        jar_csv_chunk.close()
                        csv_file.close()
                        count = number_of_split
                count = 0
                csv_files = glob.glob('./temp_split/*.csv')
                Parallel(n_jobs = number_of_split, verbose = 1)(delayed(classification) (model, csv, pred_file) for csv in csv_files)
                for csv in csv_files:
                    os.remove(csv)
            end = datetime.now() - start
            print("Elapsed time : {}".format(end))
            os.rmdir('temp_split')
        except StopIteration:
            print('Empty file.')

def order_csv(pred_csv, input_csv, csv_delimiter, id, dictionary):
    temp = pd.read_csv(pred_csv, delimiter = ",")
    temp = temp.sort_values(by=['ID'])
    change = False
    df = {}
    temp_input = None
    if id is not None:
        change = True
        temp_input = pd.read_csv(input_csv, delimiter = csv_delimiter)
        df.update({'ID': temp_input.iloc[:, 0:1].values.ravel()})
    if dictionary is not None:
        change = True
        if temp_input is None:
            temp_input = pd.read_csv(input_csv, delimiter = csv_delimiter)
        df.update({'TEXT': temp_input.iloc[:, -1:].values.ravel()})
    if not change:
        temp.to_csv(pred_csv, index = False)
    else:
        print(temp.head(5))
        temp_predicted = temp.iloc[:, -1:].values.ravel()
        print(len(temp_predicted))
        print(temp_predicted[1])
        print(temp_predicted[2])
        df.update({'PREDICTED': temp_predicted})
        temp = pd.DataFrame(df)
        temp.to_csv(pred_csv, index = False)
            
def main():
    parser = argparse.ArgumentParser(description = "Classification")
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
        print("Too many input file.")
        sys.exit(1)
    elif len(args.input) < 2:
        print("Error")
        sys.exit(1)

    try:
        check_csv(jar_csv)
        check_csv(input_csv)
        create_prediction_file(args.output)
        with open(input_csv, 'r+', newline = '') as csv_file:
            try: 
                csv_delimiter = csv.Sniffer().sniff(csv_file.read(2048)).delimiter
                print("delimiter = '{}'".format(csv_delimiter))
            except csv.Error as e:
                print("Can't find delimiter ';' used.")
                csv_delimiter = ';'
            csv_file.seek(0)
            csv_file_reader =  csv.reader(csv_file, delimiter = csv_delimiter)
            header = next(csv_file_reader)
            if len(header) == 2:
                if header[0].lower() == 'id' and header[1].lower() == 'text':
                    check_csv(jar_csv)
                    create_split_and_predict(jar_csv, model, args.chunk_size, args.number_of_jobs, args.output)
                    if args.documents == "no":
                        order_csv(args.output, input_csv, csv_delimiter, True, False)
                    else:
                        order_csv(args.output, input_csv, csv_delimiter, True, True)
                else:
                    print("Wrong header name for csv file") 
            elif len(header) == 1:
                if(header[0].lower() == 'text'):
                    check_csv(jar_csv)
                    create_split_and_predict(jar_csv, model, args.chunk_size, args.number_of_jobs, args.output)
                    if args.documents == "no":
                        order_csv(args.output, input_csv, csv_delimiter, False, False)
                    else:
                        order_csv(args.output, input_csv, csv_delimiter, False, True)
                else:
                    print("Wrong header for csv file")
            elif len(header) == 0:
                print("Empty csv file.")
            else:
                print("Too many header inside csv file")
        csv_file.close()
    except OSError as e:
        print(e)
        sys.exit(1)


if __name__ == '__main__':
    main()
