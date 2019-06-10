from sklearn.preprocessing import LabelEncoder
from liblinearutil import *

class Train():
    
    def __init__(self, jobs_number, solver_value, c_value, model_name):
        self.jobs_number = jobs_number
        self.solver_value = solver_value
        self.c_value = c_value
        self.model_name = model_name

    def train_model(self, X_train, X_test, y_train, y_test):
        le = LabelEncoder()
        le.fit(y_train)
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)
    
        if self.solver_value == 4 or self.solver_value == 7:
            parameters = "-s {} -c {} -B 1 -q".format(self.solver_value, self.c_value)
        else:
            parameters = "-s {} -n {} -c {} -B 1 -q".format(self.solver_value, self.jobs_number, self.c_value)
        param = parameter(parameters)
        prob = problem(y_train, X_train)
        model = train(prob, param)
        
        p_label, p_acc, p_val = predict(y_test, X_test, model)

        #Convert predicted value from float to int
        y_pred = [int(label) for label in p_label]
        y_test = le.inverse_transform(y_test)
        y_pred = le.inverse_transform(y_pred)
        
        save_model(f"./{self.model_name}.model", model)
        return y_pred