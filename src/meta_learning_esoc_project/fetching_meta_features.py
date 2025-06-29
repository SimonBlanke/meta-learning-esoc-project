import pandas as pd
import numpy as np
from sklearn.base import is_classifier, is_regressor

class MetaFeatures:
    def __init__(self):
        pass
    
    def dataset_meta_features(dataset_df):
        """Calculate the number of features, categorical features, and number of numerical features in the dataset"""
        numerical_df = dataset_df.select_dtypes(include=['int64', 'float64'])
        num_features = len(dataset_df.columns)
        categorical_features = len(dataset_df.select_dtypes(include=['object']).columns)
        numerical_features = len(numerical_df.columns)
        mean_correlation = numerical_df.corr().mean().mean()
        return num_features, categorical_features, numerical_features, mean_correlation
    
    def model_meta_features(model):
        """Fetch meta features from a model"""
        model_hyperparams = model.get_params()
        num_hyperparams = len(model_hyperparams)

        # check whether the model is a regressor or classifier or both
        is_clfr = is_classifier(model)
        is_regr = is_regressor(model)

        return num_hyperparams, is_clfr, is_regr