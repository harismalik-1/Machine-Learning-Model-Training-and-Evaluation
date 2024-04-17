import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.num_samples = None
        self.x_num_coords = None
        self.W_matrix = None
        self.num_categs = None

    def softmax_predictor_vect(self, x_vect):
        # evaluated by the softmax formula
        # return a vector y-hat and the index(category)
        y_pred_vect = []
        list_of_numerators = np.exp(np.dot(self.W_matrix, x_vect))
        denom = np.sum(list_of_numerators)
        y_pred_vect = list_of_numerators * denom
        
        # assigning an actual category
        category_int = np.argmax(np.array(y_pred_vect))
        y_pred_vect_result = np.zeros(self.num_categs)
        y_pred_vect_result[category_int] = 1
        return y_pred_vect_result, category_int

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        self.num_samples = len(training_data)   # N
        self.x_num_coords = len(training_data[0])   # D
        self.num_categs = max(training_labels)+1  # C
        self.W_matrix = np.random.rand(self.num_categs, self.x_num_coords+1)

        # add a column of 1s
        new_col = np.ones((self.num_samples, 1))
        self.biased_training_data = np.concatenate((new_col, training_data), axis=1)


        # gradient descent
        for _ in range(self.max_iters):
            gradR = np.zeros((self.num_categs, self.x_num_coords+1))
            for i in range(self.num_samples):
                # create the actual y vector and get x vector
                y_vect = np.zeros(self.num_categs)
                y_vect[training_labels[i]] = 1
                x_vect = np.array(self.biased_training_data[i])
                #geberate the gradient matrix
                error = np.array(self.softmax_predictor_vect(self.biased_training_data[i])[0] - y_vect)
                gradR += np.outer(error, x_vect)
            old_W_matrix = self.W_matrix
            self.W_matrix -= self.lr*gradR
            if abs(np.linalg.norm(old_W_matrix-self.W_matrix)) < 0.001:
                break
        
        pred_labels = []
        for x in self.biased_training_data:
            pred_labels.append(self.softmax_predictor_vect(x)[1])

        return np.array(pred_labels)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        pred_labels = []
        # add a column of 1s
        new_col = np.ones((len(test_data), 1))
        biased_test_data = np.concatenate((new_col, test_data), axis=1)

        for x in biased_test_data:
            pred_labels.append(self.softmax_predictor_vect(x)[1])

        return np.array(pred_labels)
