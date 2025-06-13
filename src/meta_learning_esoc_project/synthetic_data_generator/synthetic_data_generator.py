import random

from sklearn.datasets import make_regression

from .synthetic_data_parameters import dataset_dict


class SyntheticData:
    def __init__(self) -> None:
        pass

    def generate_random(self):
        return make_regression(**random.choice(list(dataset_para.values())))
