import numpy as np

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind
        self.training_data = None
        self.training_labels = None

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """

        self.training_data = training_data
        self.training_labels = training_labels
        pred_labels = self.predict(training_data)
        return pred_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """

        #helper functio to 
        def create_label_to_list(label_to_list, nearest_neighbors, distances,label):
            for idx in nearest_neighbors:
                if self.training_labels[idx] == label:
                    if self.training_labels[idx] not in label_to_list:
                        label_to_list[self.training_labels[idx]] = np.mean(np.array([distances[idx]]))
                    else:
                        label_to_list[self.training_labels[idx]] = np.mean(np.append(label_to_list[self.training_labels[idx]], distances[idx]))
            return label_to_list

        # model starts here
        predictions = []
        for sample in test_data:
            distances = [np.sqrt(np.sum((sample - x)**2)) for x in self.training_data]
            nearest_neighbors = np.argsort(distances)[:self.k]
            nearest_labels = [self.training_labels[i] for i in nearest_neighbors]

            label_to_list ={}
            label_count = {}
            for label in set(self.training_labels):
                label_to_list = create_label_to_list(label_to_list, nearest_neighbors, distances, label)
                label_count[label] = nearest_labels.count(label)

            if len(label_count.values()) != len(set(self.training_labels)):
                predicted_label = max(set(nearest_labels), key=nearest_labels.count)
            else:
                predicted_label = list(label_to_list.keys())[0]
                for label in label_to_list:
                    if label_count[label] == max(label_count.values()) and label_to_list[label] <= label_to_list[predicted_label]:
                        predicted_label = label

            predictions.append(predicted_label)
        return np.array(predictions)

