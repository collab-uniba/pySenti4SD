from time import time
from collections import OrderedDict

from sklearn.preprocessing import LabelEncoder
from liblinearutil import *

class Tuning():    

    def __init__(self, jobs_number, solvers_value_file):
        self.solvers = {
            "L2-regularized logistic regression (primal)" : 0,
            "L2-regularized L2-loss support vector classification (dual)" : 1,
            "L2-regularized L2-loss support vector classification (primal)": 2,
            "L2-regularized L1-loss support vector classification (dual)": 3,
            "support vector classification by Crammer and Singer": 4,
            "L1-regularized L2-loss support vector classification": 5,
            "L1-regularized logistic regression": 6,
            "L2-regularized logistic regression (dual)": 7
        }
        self.C_VALUE = [0.01, 0.05, 0.10, 0.20, 0.25, 0.50, 1, 2, 4, 8]
        self.jobs_number = jobs_number
        self.model_name = model_name
        self.__load_solvers_value()

    def __load_solvers_value(self, solvers_value_file):
        with open(solvers_value_file, 'r') as sf:
            for line in sf:
                if line.rtrip('\n') is not self.solvers.keys():
                    del self.solvers[line]
        sf.close()

    def __encode_label(self, y_train, y_test):
        le = LabelEncoder()
        le.fit(y_train)
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)

    def __create_perfomance_file(self, output_dir, perfomance_dict):
        with open(f"{output_dir}\{perfomance_dict['Solver Name']}", 'w') as sf:
            for value in perfomance_dict.keys():
                sf.write(f"{value}: {perfomance_dict[value]}\n")
        sf.close()    
        
    def __train_and_predict(self, X_train, X_test, y_train, y_test, solver_value, c_value):
        if solver_value == 4 or solver_value == 7:
            parameters = "-s {} -c {} -B 1 -q".format(solver_value, c_value)
        else:
            parameters = "-s {} -n {} -c {} -B 1 -q".format(solver_value, self.jobs_number, c_value)
        param = parameter(parameters)
        time_start = time()
        model = train(self.prob, param)
        training_time = time() - time_start
        
        time_start = time()
        p_label, p_acc, p_val = predict(y_test, X_test, model)
        test_time = time() - time_start

        #Convert predicted value from float to int
        y_pred = [int(label) for label in p_label]
        
        accuracy = accuracy_score(self.y_test, y_pred)
        
        return (training_time, test_time, accuracy)
        
        
    def tuning_parameter(self, X_train, X_test, y_train, y_test, output_dir):
        self.__encode_label(y_train, y_test)
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
                    parameters = "-s {} -n {} -c {} -v 10 -B 1 -q".format(solver_value, jobs_number, c_value)
                param = parameter(parameters)
                cv_accuracy = train(self.prob, param)
                if cv_accuracy > best_cv_accuracy:
                    best_c_value = c_value
                    best_cv_accuracy = cv_accuracy
                    best_s_value = solver_value
                    best_solber_name = solver_name
                if cv_accuracy > current_cv_accuracy:
                    current_cv_accuracy = cv_accuracy
                    current_c_value = c_value
            tuning_time = time() - time_start
            #Training current model for testing
            training_time, test_time, accuracy = self.__train_and_predict(X_train, X_test, y_train, y_test, solver_value, current_c_value)
            perfomance_dict = {
                "Solver Name": solver_name,
                "C value": current_c_value,
                "Tuning time": tuning_time,
                "Training time": training_time,
                "Test time": test_time,
                "Accuracy": accuracy
            }
            self.__create_perfomance_file(output_dir, perfomance_dict)
            current_cv_accuracy = 0
            current_c_value = 0
            print("\n")
    
        training_time, test_time, accuracy = self.__train_and_predict(X_train, X_test, y_train, y_test, solver_value, current_c_value)
        perfomance_dict = {
            "Solver Name": best_solver_name,
            "C value": current_c_value,
            "Tuning time": tuning_time,
            "Training time": training_time,
            "Test time": test_time,
            "Accuracy": accuracy
        }
        self.__create_perfomance_file(output_dir, perfomance_dict)

        return best_s_value, best_c_value
