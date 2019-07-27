from sklearn.ensemble import IsolationForest 
import numpy as np

def ExtraIsolationTrees(X, random_state = None):
    '''
    take as input a dataframe
    '''
    resultats = []

    for variable in range(len(X)) :
        
        X_one = X[:,variable]
        ifor = IsolationForest(behaviour='new', random_state=random_state)
        ifor.fit(X_one)

        score = abs(ifor.score_samples(X_one))

        resultats.append(np.max(score))

    return resultats



def 
