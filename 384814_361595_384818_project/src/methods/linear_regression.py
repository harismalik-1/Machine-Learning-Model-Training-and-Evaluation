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
        self.weights = None

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
        I = np.eye(D)  # Identity matrix
        I[0, 0] = 0  # Do not regularize the bias term

        # Append a column of ones to include an intercept in the model
        X_bias = np.hstack([np.ones((N, 1)), training_data])

        # Closed-form solution for the weights
        XTX = X_bias.T @ X_bias + self.lmda * I
        XTy = X_bias.T @ training_labels
        self.weights = np.linalg.solve(XTX, XTy)

        # Predict on training data to provide immediate feedback on fit
        return self.predict(training_data)
   

def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,regression_target_size)
        """

        N = test_data.shape[0]
        # Include an intercept term
        test_bias = np.hstack([np.ones((N, 1)), test_data])
        pred_labels = test_bias @ self.weights  # Matrix multiplication for predictions

        return pred_labels
