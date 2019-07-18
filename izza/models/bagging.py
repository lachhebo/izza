import numpy as np 
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator,clone
from sklearn.linear_model import LinearRegression

class ClusterBag(BaseEstimator):

    def __init__(self, estimatorcluster = None, estimatormodel = None):
        '''
        parameter : the estimator to use in the clustering part 
        the model to use in the supervised fit 
        parameter for both of them 
        '''

        if estimatorcluster == None :
            self.estimatorcluster = KMeans(n_clusters=3)
        else : 
            self.estimatorcluster = estimatorcluster

        if estimatormodel == None : 
            self.estimatormodel = LinearRegression()
        else :
            self.estimatormodel = estimatormodel

        
        self.estimators_ = {}

        self.is_fitted_cluster = 0
        self.is_fitted = 0

        
    def fit(self, X, y):
        '''
        fit the cluster model then the estimators models
        '''

        self.fit_estimator_cluster(X)
        self.fit_estimators(X,y,self.estimatormodel)

        self.is_fitted_cluster = 1

        return self

    def fit_estimator_cluster(self, X, y = None):
        '''
        Fit the cluster model
        '''

        ## Fit the cluster estimator : 

        self.labels_ = self.estimatorcluster.fit_predict(X)
        self.estimators_ = {}

        self.is_fitted_cluster = 1

    
    def fit_estimator(self,X, y, label_cluster,estimatorprediction):
        '''
        fit an estimator on the dataset related to his cluster
        '''

        ## check if estimatorcluster is fitted 

        if self.is_fitted_cluster == 0 :
            raise Exception('You need to first fit the cluster estimator ')

        ## Create a clone of our model 

        estimator =  clone(estimatorprediction)
        
        ## Range our model cleanly

        ind = []
        ind = np.where(self.labels_ == label_cluster)[0]
        self.estimators_[label_cluster] = estimator

        ## Fit our model 

        self.estimators_[label_cluster].fit(X[ind],y[ind])

    def fit_estimators(self,X,y, estimatorprediction): 
        '''
        fit all models for all clusters 
        '''

        ## check if estimatorcluster is fitted 

        if self.is_fitted_cluster == 0 :
            raise Exception('You need to first fit the cluster estimator ')

        ## Fit each cluster with our model

        for label in np.unique(self.labels_):
            self.fit_estimator(X, y,label,estimatorprediction)

        self.is_fitted = 1

    
    def predict(self,X):
        '''
        make a prediction by first finding the cluster and the model related to X 
        and then applying predict on the correct model
        '''

        ## Check everything is fitted 

        if self.is_fitted == 0 : 
            raise Exception('You need to first fit the cluster bag estimator')


        label = self.estimatorcluster.predict(X)

        
        prediction = []
        for i in range(0,len(label)):
            result = self.estimators_[label[i]].predict(X[i].reshape(1,-1))
            prediction.append(result)
        
        return np.array(prediction)


    def get_params(self, deep=True):
        out = {}

        out["estimatorcluster"] = self.estimatorcluster
        out["estimatormodel"] = self.estimatorcluster

        for model in self.estimators_ :
            out["estimator"+str(model)] = self.estimators_[model]


        for key in self.estimatorcluster.get_params() :
            out["estimatorcluster__"+str(key)] = self.estimatorcluster.get_params()[key]

        for  key in  self.estimatormodel.get_params() :
            out["estimatormodel__"+str(key)] =  self.estimatormodel.get_params()[key]
            
        for model in self.estimators_ : 
            for key in self.estimators_[model].get_params() :
                out["estimator"+str(model)+"__"+str(key)] = self.estimators_[model].get_params()[key]

        return out

