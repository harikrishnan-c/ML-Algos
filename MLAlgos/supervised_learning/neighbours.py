import numpy as np
import sys
from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from metrics.distance_metrics import DistanceMetric


# from ..metrics.distance_metrics import DistanceMetric


class KNeighboursClassifier:
    def __init__(self, n_neighbors=5, weights='uniform', p=2, metric='minkowski'):
        self.y = None
        self.x = None
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.metric = metric
        self.classes_ = None
        self.n_features_in_ = 0
        self.n_samples_fit = 0
        # TODO: Implement KD Tree and Ball Tree
        self.algorithm = 'brute'

    def fit(self, x, y):
        self.x = x
        self.y = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = x.shape[1]
        self.n_samples_fit = x.shape[0]
        # Nearest Neighbour is a Lazy Evaluation Algorithm, so during fit just training data is memorized

    def __get_neighbours(self, x_test):
        distance_met = DistanceMetric().get_metric(self.metric, self.p)
        dist = distance_met.get_distance(x_test, self.x)
        self.__best_n_classes = []
        self.__best_n_dist = []
        for dist_row in dist:
            best_n_indices = (-dist_row).argsort()[-self.n_neighbors:][::-1]
            self.__best_n_classes.append(np.array(self.y)[best_n_indices])
            self.__best_n_dist.append(dist_row[best_n_indices])
        self.__best_n_classes = np.array(self.__best_n_classes)
        self.__best_n_dist = np.array(self.__best_n_dist)

    def predict(self, x_test):
        self.__get_neighbours(x_test)
        pred = []
        if self.weights == 'uniform':
            for n_class_row in self.__best_n_classes:
                values, counts = np.unique(n_class_row, return_counts=True)
                pred.append(values[np.argmax(counts)])
        elif self.weights == 'distance':
            for n_class_row,n_best_dist in zip(self.__best_n_classes, self.__best_n_dist):
                prob_dict = {y_class: 0 for y_class in self.classes_}
                for cls, dist in zip(n_class_row,n_best_dist):
                    if dist != 0:
                        prob_dict[cls] += (1/dist)
                    else:
                        prob_dict[cls] += 1
                pred.append(max(prob_dict, key=prob_dict.get))
        return np.array(pred)

    def predict_proba(self, x_test):
        self.__get_neighbours(x_test)
        pred = []
        for n_class_row in self.__best_n_classes:
            values, counts = np.unique(n_class_row, return_counts=True)
            prob_dict = {y_class: 0 for y_class in self.classes_}
            for value, count in zip(values, counts):
                prob_dict[value] = count / self.n_neighbors
            pred.append(list(prob_dict.values()))
        return np.array(pred)
