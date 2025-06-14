import pandas as pd
import numpy as np

def fetch_meta_features(dataset_df):
    """Calculate the number of features, categorical features, and number of numerical features in the dataset"""
    numerical_df = dataset_df.select_dtypes(include=['int64', 'float64'])
    num_features = len(dataset_df.columns)
    categorical_features = len(dataset_df.select_dtypes(include=['object']).columns)
    numerical_features = len(numerical_df.columns)
    mean_correlation = numerical_df.corr().mean().mean()
    return num_features, categorical_features, numerical_features, mean_correlation