import os
import sys
import logging
import copy
import pandas as pd
import numpy as np
import tensorflow as tf

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
logger = logging.getLogger()

EPSILON = sys.float_info.min


class DataHandler:
    """

    """
    def __init__(self, input_path: str, uncertainty_path: str, output_path: str, features: list = None,
                 index_col: str = None, drop_col: list = None):
        """
        Check, load and prep the input and output data paths/files.
        :param input_path: The path to the concentration data file
        :param uncertainty_path: The path to the uncertainty data file
        :param output_path: The path to the directory where the output files are written to
        """
        self.input_path = input_path
        self.uncertainty_path = uncertainty_path
        self.output_path = output_path
        self.error = False
        self.error_list = []

        self.input_data = None
        self.uncertainty_data = None

        self.metrics = None

        self.input_data_processed = None
        self.uncertainty_data_processed = None

        self.index_col = index_col
        self.drop_col = drop_col

        self.features = features
        self.min_values = None
        self.max_values = None

        self.input_dataset = None

        self.features = None
        self.metadata = {}

        self._check_paths()
        self._load_data()

    def _check_paths(self):
        """
        Check all data paths for errors
        :return: None
        """
        if not os.path.exists(self.input_path):
            self.error = True
            self.error_list.append(f"Input file not found at {self.input_path}")
        if not os.path.exists(self.uncertainty_path):
            self.error = True
            self.error_list.append(f"Uncertainty file not found at {self.uncertainty_path}")
        if not os.path.exists(self.output_path):
            try:
                os.mkdir(self.output_path)
            except FileNotFoundError:
                self.error = True
                self.error_list.append(f"Unable to find or create output directory at {self.output_path}")
        if self.error:
            logger.error("File Errors: " + ", ".join(self.error_list))
            exit()
        else:
            logger.info("Input and output configured successfully")

    def _set_dataset(self, data, uncertainty):
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            data = data.to_numpy()
        if isinstance(uncertainty, pd.DataFrame) or isinstance(uncertainty, pd.Series):
            uncertainty = uncertainty.to_numpy()

        data[data < 0] = EPSILON
        uncertainty[uncertainty < 0] = EPSILON

        self.input_data_processed = data.astype("float32")
        self.uncertainty_data_processed = uncertainty.astype("float32")
        self.input_dataset = (tf.constant(self.input_data_processed), tf.constant(self.uncertainty_data_processed))

    def _load_data(self):
        """
        Loads the input and uncertainty data
        :return: None
        """
        if self.error:
            logger.warn("Unable to load data because of setup errors.")
            return
        if self.index_col is None:
            self.input_data = pd.read_csv(self.input_path)
            self.uncertainty_data = pd.read_csv(self.uncertainty_path)
        else:
            self.input_data = pd.read_csv(self.input_path, index_col=self.index_col)
            self.uncertainty_data = pd.read_csv(self.uncertainty_path, index_col=self.index_col)
        self.features = list(self.input_data.columns) if self.features is None else self.features

        if self.drop_col is not None:
            _input_data = self.input_data.drop(self.drop_col, axis=1)
            _uncertainty_data = self.uncertainty_data.drop(self.drop_col, axis=1)
        else:
            _input_data = self.input_data
            _uncertainty_data = self.uncertainty_data
        self._set_dataset(_input_data, _uncertainty_data)

        # self.min_values = self.input_data.min(axis=0).combine(self.uncertainty_data.min(axis=0), min)
        # self.max_values = self.input_data.max(axis=0).combine(self.uncertainty_data.max(axis=0), max)
        self.min_values = _input_data.min(axis=0)
        self.max_values = _input_data.max(axis=0)

        c_df = _input_data.copy()
        u_df = _uncertainty_data.copy()

        min_con = c_df.min()
        p25 = c_df.quantile(q=0.25, numeric_only=True)
        median_con = c_df.median(numeric_only=True)
        p75 = c_df.quantile(q=0.75, numeric_only=True)
        max_con = c_df.max()

        d = (c_df - u_df).divide(u_df, axis=0)
        mask = c_df <= u_df
        d.mask(mask, 0, inplace=True)
        sn = (1 / d.shape[0]) * d.sum(axis=0)

        categories = ["Strong"] * d.shape[1]

        self.metrics = pd.DataFrame(
            data={"Category": categories, "S/N": sn, "Min": min_con, "25th": p25, "50th": median_con, "75th": p75,
                  "Max": max_con})

    def remove_outliers(self, quantile: float = 0.8, drop_min: bool = True, drop_max: bool = False):
        """
        Remove outliers from the input dataset
        :param quantile:
        :param drop_min:
        :param drop_max:
        :return:
        """
        if self.error:
            logger.warn("Unable to process input data because of setup errors")
            return
        temp_data = copy.copy(self.input_data)
        temp_uncertainty = copy.copy(self.uncertainty_data)
        max_q_value = temp_data.quantile(quantile, axis=0)
        min_q_value = temp_data.quantile(1.0-quantile, axis=0)

        if drop_max:
            max_outlier_mask = temp_data.ge(max_q_value)
            temp_data = temp_data.mask(max_outlier_mask, np.nan)
            temp_uncertainty = temp_uncertainty.mask(max_outlier_mask, np.nan)
            self.metadata["max_outlier_values"] = list(max_q_value)
            self.metadata["max_outlier_quantile"] = quantile
        if drop_min:
            min_outlier_mask = temp_data.le(min_q_value)
            temp_data = temp_data.mask(min_outlier_mask, np.nan)
            temp_uncertainty = temp_uncertainty.mask(min_outlier_mask, np.nan)
            self.metadata["min_outlier_values"] = list(min_q_value)
            self.metadata["min_outlier_quantile"] = 1.0 - quantile
        temp_data = temp_data.dropna(axis=0)
        temp_uncertainty = temp_uncertainty.dropna(axis=0)
        self._set_dataset(temp_data, temp_uncertainty)
        self.min_values = temp_data.min(axis=0).combine(temp_uncertainty.min(axis=0), min)
        self.max_values = temp_data.max(axis=0).combine(temp_uncertainty.max(axis=0), max)
        self.min_values[self.min_values <= 0] = 0.0
        # self.min_values = temp_data.min(axis=0)
        # self.max_values = temp_data.max(axis=0)
        logger.info(f"Removed outliers for quantile: {quantile}, min values: {drop_min}, max values: {drop_max}")
        logger.info(f"Original row count: {self.input_data.shape[0]}, Updated row count: {temp_data.shape[0]}")

    def scale(self, data=None, min_values=None, max_values=None, min_value=1e-10):
        """
        Min/max scaling
        :return:
        """
        _min_values = (self.min_values.to_numpy() if min_values is None else min_values) + min_value
        _max_values = (self.max_values.to_numpy() if max_values is None else max_values) + min_value
        _data = self.input_data_processed if data is None else data
        scaled_data = (_data - _min_values) / (_max_values - _min_values)
        if data is None:
            scaled_uncertainty = (self.uncertainty_data_processed - _min_values) / (_max_values - _min_values)
            self._set_dataset(scaled_data, scaled_uncertainty)
        else:
            return scaled_data

    def remove_noisy(self, max_sn=1.0):
        _input_data = self.input_data.copy()
        _uncertainty_data = self.uncertainty_data.copy()
        for k, sn in self.metrics["S/N"].items():
            if sn < max_sn:
                _input_data = _input_data.drop(k, axis=1)
                _uncertainty_data = _uncertainty_data.drop(k, axis=1)
        self._set_dataset(_input_data, _uncertainty_data)
