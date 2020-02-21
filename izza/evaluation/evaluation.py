import numpy as np
from sklearn.metrics import recall_score, precision_score


def fun_precision(pourcentage, predicted_proba, X, y):
    '''
    evaluate the precision on the dataset X

    Parameters
    ----------

    pourcentage : the percentage of the population to take into account when
        calculating the precision and the recall

    predicted_proba : the probability of being a fraudster for each observation

    X : the target variable
        numpy array

    y : the explaining features
        numpy array

    Return
    -------

    precision_score : the precision on the dataset

    '''

    ratio = -pourcentage * int(X.shape[0] / 100)
    ind = np.argpartition(predicted_proba, ratio)[ratio:]
    predicted = np.zeros(y.shape[0])
    predicted[ind] = 1
    return precision_score(y_true=y, y_pred=predicted)


def fun_recall(pourcentage, predicted_proba, X, y):
    '''
    evaluate the recall on the dataset X

    Parameters
    ----------

    pourcentage : the percentage of the population to take into account
        when calculating the precision and the recall

    predicted_proba : the probability of being a fraudster
        for each observation

    X : the target variable
        numpy array

    y : the explaining features
        numpy array

    Return
    -------

    recall_score : the recall on the dataset

    '''

    ratio = -pourcentage * int(X.shape[0] / 100)
    ind = np.argpartition(predicted_proba, ratio)[ratio:]
    predicted = np.zeros(y.shape[0])
    predicted[ind] = 1
    return recall_score(y_true=y, y_pred=predicted)
