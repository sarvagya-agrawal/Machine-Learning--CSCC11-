"""
CSCC11 - Introduction to Machine Learning, Fall 2020, Assignment 3
B. Chan, Z. Zhang, D. Fleet
"""

import numpy as np

from logistic_regression import LogisticRegression
from utils import load_pickle_dataset

def train(train_X,
          train_y,
          test_X=None,
          test_y=None,
          factor=1,
          bias=0,
          alpha_inverse=0,
          beta_inverse=0,
          num_epochs=1000,
          step_size=1e-3,
          check_grad=False,
          verbose=False):
    """ This function trains a logistic regression model given the data.

    Args:
    - train_X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional training inputs.
    - train_y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar training outputs (labels).
    - test_X (ndarray (shape: (M, D))): A NxD matrix consisting M D-dimensional test inputs.
    - test_y (ndarray (shape: (M, 1))): A N-column vector consisting M scalar test outputs (labels).

    Initialization Args:
    - factor (float): A constant factor of the randomly initialized weights.
    - bias (float): The bias value

    Learning Args:
    - num_epochs (int): Number of gradient descent steps
                        NOTE: 1 <= num_epochs
    - step_size (float): Gradient descent step size
    - check_grad (bool): Whether or not to check gradient using finite difference.
    - verbose (bool): Whether or not to print gradient information for every step.
    """
    train_accuracy = 0
    # ====================================================
    # TODO: Implement your solution within the box
    # Step 1: Initialize model and initialize weights
    D = train_X.shape[1]
    K = len(np.unique(train_y))
    logistic_reg_model = LogisticRegression(D, K)
    logistic_reg_model.init_weights(factor, bias)

    # Step 2: Train the model
    logistic_reg_model.learn(train_X, train_y, num_epochs, step_size, check_grad, verbose, alpha_inverse, beta_inverse)

    # Step 3: Evaluate training performance

    train_probs = logistic_reg_model.predict(train_X)
    # ====================================================
    train_preds = np.argmax(train_probs, axis=1)
    train_accuracy = 100 * np.mean(train_preds == train_y.flatten())
    print("Training Accuracy: {}%".format(train_accuracy))

    if test_X is not None and test_y is not None:
        test_accuracy = 0
        # ====================================================
        # TODO: Implement your solution within the box
        # Evaluate test performance
        test_probs = logistic_reg_model.predict(test_X)
        # ====================================================
        test_preds = np.argmax(test_probs, axis=1)
        test_accuracy = 100 * np.mean(test_preds == test_y.flatten())
        print("Test Accuracy: {}%".format(test_accuracy))


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)

    # Support iris, generic_1, generic_2
    dataset = "generic_2"

    assert dataset in ("iris", "generic_1", "generic_2"), f"Invalid dataset: {dataset}"

    dataset_path = f"./datasets/{dataset}.pkl"
    data = load_pickle_dataset(dataset_path)

    train_X = data['train_X']
    train_y = data['train_y']
    test_X = test_y = None
    if 'test_X' in data and 'test_y' in data:
        test_X = data['test_X']
        test_y = data['test_y']

    # Hyperparameters
    # NOTE: This is definitely not the best way to pass all your hyperparameters.
    #       We can usually use a configuration file to specify these.
    factor = 1
    bias = 0
    alpha_inverse = 0
    beta_inverse = 0
    num_epochs = 1000
    step_size = 1e-3
    check_grad = True
    verbose = False

    train(train_X=train_X,
          train_y=train_y,
          test_X=test_X,
          test_y=test_y,
          factor=factor,
          bias=bias,
          num_epochs=num_epochs,
          step_size=step_size,
          check_grad=check_grad,
          verbose=verbose)
