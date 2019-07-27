import numpy as np 
from sklearn.metrics import recall_score, precision_score


def score_precision(i,scores,y_test):
    "Assses the value of the precision at the percentage i"
    ind = np.argpartition(scores, -int((i/100)*len(y_test)))[-int((i/100)*len(y_test)):]
    pred = np.zeros(len(y_test))
    pred[ind] = 1
    return precision_score(y_test, pred)
    
def score_recall(i,scores,y_test):
    "Assses the value of the recall at the percentage i"
    ind = np.argpartition(scores, -int((i/100)*len(y_test)))[-int((i/100)*len(y_test)):]
    pred = np.zeros(len(y_test))
    pred[ind] = 1
    return recall_score(y_test, pred)
