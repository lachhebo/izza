import unittest
import numpy as np
from izza.models import ClusterBag 
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston


class BaggingTest(unittest.TestCase):


    def test_fit_estimator_cluster(self):
        '''
        Fit the cluster model
        '''

        ## Arrange 
        boston = load_boston()
        X = boston.data
        ## Test : 

        bagging = ClusterBag()

        self.assertEqual( bagging.is_fitted_cluster, 0)

        bagging.fit_estimator_cluster(X)

        ## Assert : 

        self.assertEqual( len(np.unique(bagging.labels_)),3)
        self.assertEqual( bagging.is_fitted_cluster, 1)
        
        
    def test_fit_estimator(self):

        ## Arrange 
        boston = load_boston()
        X = boston.data
        y = boston.target

        ## Test : 

        bagging = ClusterBag()
        bagging.fit_estimator_cluster(X)
        bagging.fit_estimator(X,y, 1, LinearRegression())

        ## Assert :

        self.assertNotIn(0,bagging.estimators_)
        self.assertIsNotNone(bagging.estimators_[1])
        self.assertNotIn(2,bagging.estimators_)
        

    def test_fit_estimators(self):

        ## Arrange 
        boston = load_boston()
        X = boston.data
        y = boston.target

        ## Test : 

        bagging = ClusterBag()
        bagging.fit_estimator_cluster(X)
        bagging.fit_estimators(X,y, LinearRegression())

        ## Assert :

        self.assertIsNotNone(bagging.estimators_[0])
        self.assertIsNotNone(bagging.estimators_[1])
        self.assertIsNotNone(bagging.estimators_[2])

        


if __name__ == '__main__':
    unittest.main()