from abc import ABC, abstractmethod
import numpy as np


class DistanceMetric:
    def __init__(self):
        pass

    @staticmethod
    def get_metric(metric='minkowski', p=1):
        if metric == 'minkowski':
            return _MinkowskiDistance(p)
        elif metric == 'euclidean':
            return _EuclideanDistance()
        elif metric == 'manhattan':
            return _ManhattanDistance()
        elif metric == 'chebyshev':
            return _ChebyshevDistance()
        else:
            # TODO: Create a Custom Exception
            raise Exception('Metric not Found!!')


class _SpatialDistance(ABC):
    @abstractmethod
    def get_distance(self, x, y):
        pass



class _MinkowskiDistance(_SpatialDistance):
    def __init__(self, p):
        self.p = p

    def get_distance(self, x, y):
        dist = []
        for x_row in x:
            dist.append(np.power(np.sum(np.power(np.abs(x_row - y), self.p), axis=1), 1 / self.p))
        return np.array(dist)


class _EuclideanDistance(_MinkowskiDistance):
    def __init__(self):
        super().__init__(2)

    def get_distance(self, x, y):
        return _MinkowskiDistance(self.p).get_distance(x, y)


class _ManhattanDistance(_SpatialDistance):
    def __init__(self):
        pass

    def get_distance(self, x, y):
        dist = []
        for x_row in x:
            dist.append(np.sum(np.abs(x_row - y), axis=1))
        return np.array(dist)


class _ChebyshevDistance(_SpatialDistance):
    def __init__(self):
        pass

    def get_distance(self, x, y):
        dist = []
        for x_row in x:
            dist.append(np.max(np.abs(x_row - y), axis=1))
        return np.array(dist)
