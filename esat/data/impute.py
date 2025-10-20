import logging

import pandas as pd
import numpy as np

from esat.data.datahandler import DataHandler

logger = logging.getLogger(__name__)

sklearn_loaded = False
try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
    sklearn_loaded = True
except ImportError:
    logger.error(f"scikit-learn is not installed. Please install it to use the DataImputer class.")



class DataImputer:
    """
    Class for imputing missing values in datasets using various strategies.

    This class supports mean, median, most_frequent, KNN, and iterative imputation strategies.

    :param input_data: The input data array with missing values.
    :param uncertainty_data: The uncertainty data array with missing values.
    :param random_seed: Random seed for reproducibility (default is 42).
    :param missing_value: Value to be treated as missing (default is np.nan).
    """
    def __init__(self, input_data: np.ndarray, uncertainty_data: np.ndarray, columns: list = None, index: list = None,
                 random_seed: int = 42, missing_value: float = np.nan):
        self.input_data = input_data
        self.uncertainty_data = uncertainty_data

        self.random_seed = random_seed

        self.columns = columns if columns is not None else [f"feature {i}" for i in range(input_data.shape[1])]
        self.index = index if index is not None else [f"sample {i}" for i in range(input_data.shape[0])]

        self.imputation_mask = None
        self.imputed_data = None
        self.imputed_uncertainty = None

        self.strategy = None
        self.missing_value = missing_value

    def impute(self, strategy='mean', args: dict = None):
        """
        Impute missing values in the dataset using the specified strategy.

        :param strategy: The imputation strategy to use ('mean', 'median', 'most_frequent', 'knn', or 'iterative').
        :param args: Additional arguments for the imputer (e.g., n_neighbors for KNN, max_iter for IterativeImputer).
        """
        if strategy not in ['mean', 'median', 'most_frequent', 'knn', 'iterative']:
            raise ValueError(f"Invalid imputation strategy: {strategy}. Choose from 'mean', 'median', 'most_frequent', "
                             f"'knn', or 'iterative'.")
        self.strategy = strategy
        logger.info(f"Imputing missing values using strategy: {strategy}")
        V = self.input_data
        U = self.uncertainty_data
        if self.missing_value is not None:
            V = np.where(V == self.missing_value, np.nan, V)
            U = np.where(U == self.missing_value, np.nan, U)

        # Create a mask for referencing missing values
        self.imputation_mask = np.isnan(V) | np.isnan(U)

        if strategy in ['mean', 'median', 'most_frequent']:
            self._run_simple_imputer(V, U, strategy)
        elif strategy == 'knn':
            if args is None:
                args = {}
            n_neighbors = args.get('n_neighbors', 5)
            weights = args.get('weights', 'uniform')
            self._run_knn_imputer(V, U, n_neighbors, weights)
        elif strategy == 'iterative':
            if args is None:
                args = {}
            max_iter = args.get('max_iter', 10)
            tol = args.get('tol', 1e-3)
            self._run_iterative_imputer(V, U, max_iter, tol)
        else:
            raise ValueError(f"Unsupported imputation strategy: {strategy}. Supported strategies are 'mean', 'median', "
                             f"'most_frequent', 'knn', and 'iterative'.")
        # Convert imputed data back to DataFrame
        return self.imputed_data, self.imputed_uncertainty

    def _run_simple_imputer(self, V, U, strategy):
        """
        Run the SimpleImputer from scikit-learn on the provided data.

        :param U: The input data to impute.
        :param V: The uncertainty data (not used in this method).
        :param strategy: The imputation strategy to use ('mean', 'median', 'most_frequent').
        :return: Imputed data.
        """
        imputer = SimpleImputer(strategy=strategy, missing_values=np.nan)
        imputer2 = SimpleImputer(strategy=strategy, missing_values=np.nan)
        imputed_data = imputer.fit_transform(V)
        imputed_uncertainty = imputer2.fit_transform(U)
        self.imputed_data = pd.DataFrame(imputed_data, columns=self.columns, index=self.index)
        self.imputed_uncertainty = pd.DataFrame(imputed_uncertainty, columns=self.columns, index=self.index)
        return self.imputed_data, self.imputed_uncertainty

    def _run_knn_imputer(self, V, U, n_neighbors=5, weights='uniform'):
        """
        Run the KNNImputer from scikit-learn on the provided data.

        :param U: The input data to impute.
        :param V: The uncertainty data (not used in this method).
        :param n_neighbors: Number of neighbors to use for imputation.
        :param weights: Weight function used in prediction ('uniform' or 'distance').
        :return: Imputed data.
        """
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights, missing_values=np.nan)
        imputer2 = KNNImputer(n_neighbors=n_neighbors, weights=weights, missing_values=np.nan)
        imputed_data = imputer.fit_transform(V)
        imputed_uncertainty = imputer2.fit_transform(U)
        self.imputed_data = pd.DataFrame(imputed_data, columns=self.columns, index=self.index)
        self.imputed_uncertainty = pd.DataFrame(imputed_uncertainty, columns=self.columns, index=self.index)
        return self.imputed_data, self.imputed_uncertainty

    def _run_iterative_imputer(self, V, U, max_iter=10, tol=1e-3):
        """
        Run the IterativeImputer from scikit-learn on the provided data.

        :param U: The input data to impute.
        :param V: The uncertainty data (not used in this method).
        :param max_iter: Maximum number of imputation iterations.
        :param tol: Tolerance for stopping criteria.
        :return: Imputed data.
        """
        imputer = IterativeImputer(max_iter=max_iter, tol=tol, random_state=self.random_seed, missing_values=np.nan)
        imputer2 = IterativeImputer(max_iter=max_iter, tol=tol, random_state=self.random_seed, missing_values=np.nan)
        imputed_data = imputer.fit_transform(V)
        imputed_uncertainty = imputer2.fit_transform(U)
        self.imputed_data = pd.DataFrame(imputed_data, columns=self.columns, index=self.index)
        self.imputed_uncertainty = pd.DataFrame(imputed_uncertainty, columns=self.columns, index=self.index)
        return self.imputed_data, self.imputed_uncertainty
