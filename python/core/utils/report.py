from collections import OrderedDict
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score


class Report():

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def print_report(self):
        print(classification_report(self.y_true, self.y_pred))

    def get_report(self):
        return classification_report(self.y_true, self.y_pred)

    def get_micro_score(self):
        return precision_recall_fscore_support(self.y_true, self.y_pred, average='micro')

    def get_macro_score(self):
        return precision_recall_fscore_support(self.y_true, self.y_pred, average='macro')

    def get_accuracy_score(self):
        return accuracy_score(self.y_true, self.y_pred)

    def get_classes_score(self):
        unique, counts = np.unique(self.y_pred, return_counts=True)
        scores = precision_recall_fscore_support(self.y_true, self.y_pred, average=None, labels=unique)
        scores_dict = OrderedDict()
        for value in unique:
            scores_dict.update({value: []})
        for i in range(len(scores) - 1):
            for j in range(len(unique)):
                scores_dict[unique[j]].append(scores[i][j])
        scores_dict.update({"support": scores[-1]})
        return scores_dict