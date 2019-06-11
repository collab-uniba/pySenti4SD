from time import time, gmtime, strftime
from collections import OrderedDict
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from liblinearutil import *

class Tuning():    

    def __init__(self, jobs_number, solvers_value_file, output_dir):
        self.solvers = OrderedDict()
        self.solvers["L2-regularized logistic regression (primal)"] = 0
        self.solvers["L2-regularized L2-loss support vector classification (dual)"] = 1
        self.solvers["L2-regularized L2-loss support vector classification (primal)"] = 2
        self.solvers["L2-regularized L1-loss support vector classification (dual)"] = 3
        self.solvers["support vector classification by Crammer and Singer"] = 4
        self.solvers["L1-regularized L2-loss support vector classification"] = 5
        self.solvers["L1-regularized logistic regression"] = 6
        self.solvers["L2-regularized logistic regression (dual)"] = 7
        self.C_VALUE = [0.01, 0.05, 0.10, 0.20, 0.25, 0.50, 1, 2, 4, 8]
        self.output_dir = output_dir
        if solvers_value_file is None:
            self.__write_solvers_value()
        self.jobs_number = jobs_number
        self.__load_solvers_value(solvers_value_file)
        self.best_perfomance = OrderedDict()

    def __write_solvers_value(self):
        with open(f"{self.output_dir}/liblinear_solver", 'w') as sf:
            for value in self.solvers.keys():
                sf.write(f"{value}\n")
        sf.close()

    def __load_solvers_value(self, solvers_value_file):
        solvers_value_file = Path(solvers_value_file)
        if not solvers_value_file.exists():
            with solvers_value_file.open('w', encoding='utf-8') as sf:
                for key in self.solvers.keys():
                    sf.write(f"{key}\n")
            sf.close()
        with open(solvers_value_file, 'r') as sf:
            lines = []
            for line in sf:
                line = line.rstrip('\n')
                print(line)
                lines.append(line)
        sf.close()
        for key in self.solvers.keys():
            if key not in lines:
                del self.solvers[key]

    def __encode_label(self, y_train, y_test):
        le = LabelEncoder()
        le.fit(y_train)
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)
        return y_train, y_test

    def __create_perfomance_file(self, perfomance_dict):
        with open(f"{self.output_dir}/{perfomance_dict['Solver name']}", 'w') as sf:
            for value in perfomance_dict.keys():
                sf.write(f"{value}: {perfomance_dict[value]}\n")
        sf.close()    
        
    def __train_and_predict(self, X_train, X_test, y_train, y_test, solver_value, c_value):
        if solver_value == 4 or solver_value == 7:
            parameters = "-s {} -c {} -B 1 -q".format(solver_value, c_value)
        else:
            parameters = "-s {} -n {} -c {} -B 1 -q".format(solver_value, self.jobs_number, c_value)
        param = parameter(parameters)

        model = train(self.prob, param)

        p_label, p_acc, p_val = predict(y_test, X_test, model)

        #Convert predicted value from float to int
        y_pred = [int(label) for label in p_label]
        
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy

        
    def tuning_parameter(self, X_train, X_test, y_train, y_test):
        y_train, y_test = self.__encode_label(y_train, y_test)
        self.prob = problem(y_train, X_train)

        self.scores_list = []
        
        cv_accuracy = 0
        
        best_solver_name = ""
        best_cv_accuracy = 0
        best_c_value = 0
        best_s_value = 0
        
        current_cv_accuracy = 0
        current_c_value = 0

        for solver_name, solver_value in self.solvers.items():
            print(f"Tuning solver {solver_name}")
            time_start = time()
            for c_value in self.C_VALUE:
                print(f"C value: {c_value}")
                if solver_value == 4 or solver_value == 7:
                    parameters = "-s {} -c {} -v 10 -B 1 -q".format(solver_value, c_value)
                else:
                    parameters = "-s {} -n {} -c {} -v 10 -B 1 -q".format(solver_value, self.jobs_number, c_value)
                param = parameter(parameters)
                cv_accuracy = train(self.prob, param)
                if cv_accuracy > best_cv_accuracy:
                    best_c_value = c_value
                    best_cv_accuracy = cv_accuracy
                    best_s_value = solver_value
                    best_solver_name = solver_name
                if cv_accuracy > current_cv_accuracy:
                    current_cv_accuracy = cv_accuracy
                    current_c_value = c_value
            tuning_time = time() - time_start
            tuning_time = strftime("%H:%M:%S", gmtime(tuning_time))

            #Training current model for testing
            accuracy = self.__train_and_predict(X_train, X_test, y_train, y_test, solver_value, current_c_value)
            perfomance_dict = OrderedDict()
            perfomance_dict["Solver name"] = solver_name
            perfomance_dict["Best C value"] = current_c_value
            perfomance_dict["Tuning time"] = tuning_time
            perfomance_dict["Accuracy"] = accuracy
            self.__create_perfomance_file(perfomance_dict)
            current_cv_accuracy = 0
            current_c_value = 0
            print("\n")
    
        #training_time, test_time, accuracy = self.__train_and_predict(X_train, X_test, y_train, y_test, best_s_value, best_c_value)
        #self.best_perfomance = OrderedDict()
        #self.best_perfomance["Solver name"] = best_solver_name
        #self.best_perfomance["C value"] = best_c_value
        #self.best_perfomance["Tuning time"] = tuning_time
        #self.best_perfomance["Training time"] = training_time
        #self.best_perfomance["Test time"] = test_time
        #self.best_perfomance["Accuracy"] = accuracy

        return best_solver_name, best_s_value, best_c_value
