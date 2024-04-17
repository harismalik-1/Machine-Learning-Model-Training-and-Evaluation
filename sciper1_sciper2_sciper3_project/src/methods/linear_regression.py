import numpy as np
import sys

class LinearRegression(object):
    """
        Linear regressor object. 
        Note: This class will implement BOTH linear regression and ridge regression.
        Recall that linear regression is just ridge regression with lambda=0.
    """

    def __init__(self, lmda):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.lmda = lmda
        self.weights = None  # weights of the model for fitting

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,regression_target_size)
        """
        N, D = training_data.shape
        I = np.eye(D)  # Identity matrix of shape (D, D)
        
        # Regularization term, note that when lambda is 0, it becomes standard linear regression
        regularization_term = self.lmda * I
        
        # Closed-form solution
        XTX = training_data.T @ training_data + regularization_term
        XTy = training_data.T @ training_labels
        
        # Computing weights
        self.weights = np.linalg.inv(XTX) @ XTy

        return pred_labels


def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,regression_target_size)
        """
        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##

        return pred_regression_targets
