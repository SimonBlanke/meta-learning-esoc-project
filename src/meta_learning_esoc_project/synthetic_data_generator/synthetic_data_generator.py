import random

from sklearn.datasets import make_regression

from .synthetic_data_parameters import dataset_params


class SyntheticData:
    datasets: list

    def __init__(self) -> None:
        pass

    def make(self, dataset_params):
        return make_regression(dataset_params)

    def generate_random(self):
        return make_regression(**random.choice(list(dataset_params.values())))

    def add(self, dataset_params):
        self.datasets.append(make(dataset_params))
