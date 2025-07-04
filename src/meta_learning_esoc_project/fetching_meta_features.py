import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from typing import Union, Tuple

class MetaFeatures:
    def __init__(self) -> None:
        pass

    def dataset_meta_features(dataset_df: pd.DataFrame) -> Tuple[int, int, int, float]:
        """Calculate the number of features, categorical features, and number of numerical features in the dataset"""

        numerical_df = dataset_df.select_dtypes(include=['int64', 'float64'])
        num_features = len(dataset_df.columns)
        categorical_features = len(dataset_df.select_dtypes(include=['object']).columns)
        numerical_features = len(numerical_df.columns)
        mean_correlation = numerical_df.corr().mean().mean()
        return num_features, categorical_features, numerical_features, mean_correlation
    
    def model_meta_features(model: DecisionTreeClassifier) -> Tuple[int, Union[int, float], int]:
        
        """Fetch meta features from a model"""
        max_depth = model.get_params()['max_depth']
        min_samples_split = model.get_params()['min_samples_split']
        # min_samples_leaf = model.get_params()['min_samples_leaf']
        # min_weight_fraction_leaf = model.get_params()['min_weight_fraction_leaf']
        # max_features = model.get_params()['max_features']
        max_leaf_nodes = model.get_params()['max_leaf_nodes']
        # min_impurity_decrease = model.get_params()['min_impurity_decrease']
        # ccp_alpha = model.get_params()['ccp_alpha']
        # class_weight = model.get_params()['class_weight']

        return (max_depth, min_samples_split, max_leaf_nodes)