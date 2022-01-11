"""
CSCC11 - Introduction to Machine Learning, Fall 2020, Assignment 2
B. Chan, E. Franco, D. Fleet

This file specifies the hyperparameters for the two real life datasets.
Note that different hyperparameters will affect the runtime of the 
algorithm.
"""

# ====================================================
# TODO: Use Validation Set to Tune hyperparameters for the Amazon dataset
# Use Optimal Parameters to get good accuracy on Test Set
AMAZON_HYPERPARAMETERS = {
    "num_trees": 240,
    "features_percent": 0.10,
    "data_percent": 0.9,
    "max_depth": 20,
    "min_leaf_data": 50,
    "min_entropy": 0.0001,
    "num_split_retries": 10
}
# ====================================================

# ====================================================
# TODO: Use Validation Set to Tune hyperparameters for the Occupancy dataset
# Use Optimal Parameters to get good accuracy on Test Set
OCCUPANCY_HYPERPARAMETERS = {
    "num_trees": 10,
    "features_percent": 0.9,
    "data_percent": 0.5,
    "max_depth": 10,
    "min_leaf_data": 10,
    "min_entropy": 0.0001,
    "num_split_retries": 10
}
# ====================================================
