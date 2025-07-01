import math
import os
import sys
import logging
import copy
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from esat.model.recombinator import optimal_block_length
from esat.utils import min_timestep

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

EPSILON = sys.float_info.min


class DataHandler:
    """
    The class for cleaning and preparing input datasets for use in ESAT.

    The DataHandler class is intended to provide a standardized way of cleaning and preparing data from file to ESAT
    models.

    The input and uncertainty data files are specified by their file paths. Input files can be .csv or tab separated
    text files. Other file formats are not supported at this time.

    #TODO: Add additional supported file formats by expanding the __read_data function.

    Parameters
    ----------
    input_path : str
        The file path to the input dataset.
    uncertainty_path : str
        The file path to the uncertainty dataset.
        #TODO: Add the option of generating an uncertainty dataset from a provided input dataset, using a random selection of some percentage range of the input dataset cell values.
    index_col : str
        The name of the index column if it is not the first column in the dataset. Default = None, which will use
        the 1st column.
    drop_col : list
        A list of columns to drop from the dataset. Default = None.
    loc_cols : str | list
        Location information columns, such as latitude/longitude or other identifier, that are used to identify the location of the data.
    sn_threshold : float
        The threshold for the signal to noise ratio values.
    load: bool
        Load the input and uncertainty data files, used internally for load_dataframe.
    loc_metadata : dict
        Optional dictionary containing metadata about the locations in the dataset, such as latitude and longitude.
    """
    def __init__(self,
                 input_path: str,
                 uncertainty_path: str,
                 index_col: str = None,
                 drop_col: list = None,
                 loc_cols: str | list = None,
                 sn_threshold: float = 2.0,
                 load: bool = True,
                 loc_metadata: dict = None
                 ):
        """
        Constructor method.
        """
        self.input_path = input_path
        self.uncertainty_path = uncertainty_path
        self.error = False
        self.error_list = []

        self.input_data = None
        self.uncertainty_data = None

        self.sn_mask = None
        self.sn_threshold = sn_threshold

        self.metrics = None

        # Processed data that is passed to a model, dataframes are used for analysis
        self.input_data_df = None
        self.uncertainty_data_df = None
        self.input_data_processed = None
        self.uncertainty_data_processed = None

        self.index_col = index_col
        self.drop_col = drop_col
        self.loc_cols = loc_cols

        self.min_values = None
        self.max_values = None

        self.features = None
        self.metadata = {}

        self.optimal_block = None

        if load:
            self._check_paths()
            self._load_data()
            self._determine_optimal_block()

    def get_data(self):
        """
        Get the processed input and uncertainty dataset ready for use in ESAT.
        Returns
        -------
        np.ndarray, np.ndarray
            The processed input dataset and the processed uncertainty dataset as numpy arrays.
        """
        self._set_dataset()
        return self.input_data_processed, self.uncertainty_data_processed

    def set_category(self, feature: str, category: str = "strong"):
        """
        Set the S/N category for the feature, options are 'strong', 'weak' or 'bad'. All features are set to 'strong'
        by default, which doesn't modify the feature's behavior in models. Features categorized as 'weak' triples their
        uncertainty and 'bad' features are excluded from analysis.

        Parameters
        ----------
        feature : str
            The name or label of the feature.
        category : str
            The new category of the feature

        Returns
        -------
        bool
            True if the change was successful, otherwise False.
        """
        if feature is None:
            logger.error("A feature name must be provided to update feature category")
            return False
        if category not in ("strong", "weak", "bad"):
            logger.error("The feature category must be set to 'strong', 'weak', or 'bad'")
            return False
        self.metrics.loc[feature, "Category"] = category.lower()
        logger.info(f"Feature: {feature} category set to {category}")
        return True

    def split_locations(self):
        """
        When the input data has location information, this function returns splits the data and uncertainty into
        separate DataHandler instances for each location.

        Returns
        -------
        list
            A list of DataHandler instances, one for each unique location in the input data.
        """
        if self.loc_cols is None:
            logger.error("No location columns specified, cannot split locations.")
            return []

        if isinstance(self.loc_cols, str):
            self.loc_cols = [self.loc_cols]

        if not all(col in self.input_data.columns for col in self.loc_cols):
            logger.error("One or more location columns not found in input data.")
            return []

        # If loc_cols are latitude and longitude, round them for comparison
        if set(self.loc_cols) == {"latitude", "longitude"}:
            rounded_input = self.input_data.copy()
            rounded_uncertainty = self.uncertainty_data.copy()
            for col in self.loc_cols:
                rounded_input[col] = rounded_input[col].round(5)
                rounded_uncertainty[col] = rounded_uncertainty[col].round(5)
            locations = rounded_input[self.loc_cols].drop_duplicates()
        else:
            rounded_input = self.input_data
            rounded_uncertainty = self.uncertainty_data
            locations = self.input_data[self.loc_cols].drop_duplicates()

        data_handlers = []

        for _, loc in locations.iterrows():
            if set(self.loc_cols) == {"latitude", "longitude"}:
                loc_filter = (rounded_input[self.loc_cols].round(5) == loc.values).all(axis=1)
                input_data_loc = self.input_data[loc_filter]
                uncertainty_data_loc = self.uncertainty_data[loc_filter]
            else:
                loc_filter = (self.input_data[self.loc_cols] == loc.values).all(axis=1)
                input_data_loc = self.input_data[loc_filter]
                uncertainty_data_loc = self.uncertainty_data[loc_filter]

            dh = DataHandler.load_dataframe(input_df=input_data_loc, uncertainty_df=uncertainty_data_loc)
            dh.metadata['location'] = loc.to_dict()
            data_handlers.append(dh)

        return data_handlers

    def merge(self, data_handlers: list, source_labels: list):
        """
        Merge a list of DataHandler instances into this DataHandler instance.
        All instances must have the same features as the current instance.
        Adds a 'source_label' column indicating the origin of each row.

        Parameters
        ----------
        data_handlers : list
            A list of DataHandler instances to merge.
        source_labels : list
            A list of labels (str) indicating the source of each DataHandler.

        Returns
        -------
        bool
            True if merging was successful, otherwise False.
        """
        if len(data_handlers) == 0 or len(data_handlers) != len(source_labels):
            logger.error("DataHandlers and source_labels must be non-empty and of equal length.")
            return False

        # Check features match
        for dh in data_handlers:
            if list(dh.input_data.columns) != list(self.input_data.columns):
                logger.error("All DataHandlers must have the same features to merge.")
                return False

        # Add source_label column
        merged_input = []
        merged_uncertainty = []
        for dh, label in zip(data_handlers, source_labels):
            input_df = dh.input_data.copy()
            input_df['source_label'] = label
            uncertainty_df = dh.uncertainty_data.copy()
            uncertainty_df['source_label'] = label
            merged_input.append(input_df)
            merged_uncertainty.append(uncertainty_df)

        # Add current instance's data
        self_input = self.input_data.copy()
        self_input['source_label'] = 'self'
        self_uncertainty = self.uncertainty_data.copy()
        self_uncertainty['source_label'] = 'self'
        merged_input.append(self_input)
        merged_uncertainty.append(self_uncertainty)

        # Concatenate
        self.input_data = pd.concat(merged_input, ignore_index=True)
        self.uncertainty_data = pd.concat(merged_uncertainty, ignore_index=True)
        self.features = [col for col in self.input_data.columns if col != 'source_label']

        logger.info("DataHandlers merged successfully.")
        return True

    def _check_paths(self):
        """
        Check all file paths to make sure they exist.
        """
        if not os.path.isabs(self.input_path):
            logger.error(f"Input file path is not absolute: {self.input_path}")
            # self.input_path = os.path.join(ROOT_DIR, self.input_path)
        if not os.path.isabs(self.uncertainty_path):
            logger.error(f"Uncertainty file path is not absolute: {self.uncertainty_path}")
            # self.uncertainty_path = os.path.join(ROOT_DIR, self.uncertainty_path)

        if not os.path.exists(self.input_path):
            self.error = True
            self.error_list.append(f"Input file not found at {self.input_path}")
        if not os.path.exists(self.uncertainty_path):
            self.error = True
            self.error_list.append(f"Uncertainty file not found at {self.uncertainty_path}")
        if self.error:
            logger.error("File Errors: " + ", ".join(self.error_list))
            sys.exit()
        else:
            logger.info("Input and output configured successfully")

    def _set_dataset(self):
        """
        Sets the processed input and uncertainty datasets.
        """
        # Drop columns if specified
        if self.drop_col is not None:
            _input_data = copy.copy(self.input_data.drop(labels=self.drop_col, axis=1)).astype("float32")
            _uncertainty_data = copy.copy(self.uncertainty_data.drop(labels=self.drop_col, axis=1)).astype("float32")
        else:
            _input_data = copy.copy(self.input_data).astype("float32")
            _uncertainty_data = copy.copy(self.uncertainty_data).astype("float32")

        # Drop location columns if specified
        if self.loc_cols is not None:
            if isinstance(self.loc_cols, str):
                self.loc_cols = [self.loc_cols]
            _input_data = _input_data.drop(labels=self.loc_cols, axis=1)
            _uncertainty_data = _uncertainty_data.drop(labels=self.loc_cols, axis=1)

        # Drop bad category features
        bad_features = list(self.metrics.loc[self.metrics["Category"] == "bad"].index)
        for bf in bad_features:
            _input_data = _input_data.drop(labels=bf, axis=1)
            _uncertainty_data = _uncertainty_data.drop(labels=bf, axis=1)
        # Multiply the uncertainty of weak category features by 3
        weak_features = list(self.metrics.loc[self.metrics["Category"] == "weak"].index)
        for wf in weak_features:
            _uncertainty_data[wf] = _uncertainty_data[wf] * 3.0

        # Exclude loc_cols from features
        if self.loc_cols is not None:
            self.features = [col for col in _input_data.columns if col not in self.loc_cols]
        else:
            self.features = list(_input_data.columns)

        # Ensure data and uncertainty values are numeric !Important!
        for f in self.features:
            _input_data[f] = pd.to_numeric(_input_data[f])
            _uncertainty_data[f] = pd.to_numeric(_uncertainty_data[f])

        self.input_data_df = _input_data
        self.uncertainty_data_df = _uncertainty_data

        if isinstance(_input_data, pd.DataFrame) or isinstance(_input_data, pd.Series):
            _input_data = _input_data.to_numpy()
        if isinstance(_uncertainty_data, pd.DataFrame) or isinstance(_uncertainty_data, pd.Series):
            _uncertainty_data = _uncertainty_data.to_numpy()

        _input_data[_input_data == 0] = EPSILON
        _uncertainty_data[_uncertainty_data == 0] = EPSILON

        self.input_data_processed = _input_data.astype("float32")
        self.uncertainty_data_processed = _uncertainty_data.astype("float32")
        self._determine_optimal_block(input_data=_input_data)

    def _read_data(self, filepath, index_col=None):
        """
        Read in a data file into a pandas dataframe.

        Parameters
        ----------
        filepath : str
            The path to the data file.
        index_col : str
            The index column of the dataset.

        Returns
        -------
        pd.DataFrame
            If the file successfully loads the function returns a pd.DataFrame otherwise it will exit.

        """
        if ".csv" in filepath:
            if index_col:
                data = pd.read_csv(filepath, index_col=index_col)
            else:
                data = pd.read_csv(filepath)
        elif ".txt" in filepath:
            if index_col:
                data = pd.read_table(filepath, index_col=index_col, sep="\t")
            else:
                data = pd.read_table(filepath, sep="\t")
            # data.dropna(inplace=True)
        else:
            logger.warning("Unknown file type provided.")
            sys.exit()
        return data

    def _load_data(self, existing_data: bool = False):
        """
        Loads the input and uncertainty data from files.
        """
        if self.error:
            logger.warning("Unable to load data because of setup errors.")
            return
        if not existing_data:
            self.input_data = self._read_data(filepath=self.input_path, index_col=self.index_col)
            self.uncertainty_data = self._read_data(filepath=self.uncertainty_path, index_col=self.index_col)
            self.features = list(self.input_data.columns) if self.features is None else self.features

        self.min_values = self.input_data.min(axis=0)
        self.max_values = self.input_data.max(axis=0)

        c_df = self.input_data.copy()
        u_df = self.uncertainty_data.copy()

        min_con = c_df.min()
        p25 = c_df.quantile(q=0.25, numeric_only=True)
        median_con = c_df.median(numeric_only=True)
        p75 = c_df.quantile(q=0.75, numeric_only=True)
        max_con = c_df.max()

        d = (c_df - u_df).divide(u_df, axis=0)
        mask = c_df <= u_df
        d.mask(mask, 0, inplace=True)
        sn = (1 / d.shape[0]) * d.sum(axis=0)

        categories = ["strong"] * d.shape[1]

        self.metrics = pd.DataFrame(
            data={"Category": categories, "S/N": sn, "Min": min_con, "25th": p25, "50th": median_con, "75th": p75,
                  "Max": max_con})

    def _determine_optimal_block(self, input_data=None):
        """
        Runs the recombinator code to obtain the optimal block size for Bootstrap based upon the Politis and White 2004
        algorithm. https://web.archive.org/web/20040726091553id_/http://1cj3301.ucsd.edu:80/hwcv-093.pdf

        Sets the self.optimal_block parameter by taking the average value of b_star_cb of each feature.
        """
        if input_data is None:
            input_data = self.input_data.to_numpy()
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                optimal_blocks = optimal_block_length(input_data)
                optimal_block = []
                for opt in optimal_blocks:
                    optimal_block.append(opt.b_star_cb)
                self.optimal_block = math.floor(np.mean(optimal_block))
        except ValueError as e:
            self.optimal_block = int(input_data.shape[1]/5)
            logger.error(f"Unable to determine optimal block size. Setting default to {self.optimal_block}")

    def plot_data_uncertainty(self, show: bool = True, include_menu: bool = True, feature_idx: int = None):
        """
        Create a plot of the data vs the uncertainty for a specified feature, with a dropdown menu for feature selection.
        """
        if self.input_data is None or self.uncertainty_data is None:
            logger.error("Input or uncertainty data is not loaded.")
            return

        if not include_menu and feature_idx is not None:
            features = [self.input_data.columns[feature_idx]]
        else:
            features = self.input_data.columns
        du_plot = go.Figure()
        buttons = []

        for feature_idx, feature_label in enumerate(features):
            feature_data = self.input_data[feature_label]
            feature_uncertainty = self.uncertainty_data[feature_label]

            # Add traces for each feature (initially hidden)
            du_plot.add_trace(
                go.Scatter(
                    x=feature_data,
                    y=feature_uncertainty,
                    mode='markers',
                    name=feature_label,
                    visible=(feature_idx == 0) or not include_menu,
                )
            )

            # Create a button for each feature
            buttons.append(
                dict(
                    label=feature_label,
                    method="update",
                    args=[
                        {"visible": [i == feature_idx for i in range(len(features))]},
                        {"title.text": f"Concentration/Uncertainty Scatter Plot - {feature_label}"}
                    ]
                )
            )
        if include_menu:
        # Add dropdown menu
            du_plot.update_layout(
                updatemenus=[
                    dict(
                        type="dropdown",
                        direction="down",
                        buttons=buttons,
                        showactive=True
                    )
                ])
        du_plot.update_layout(
            title=f"Concentration/Uncertainty Scatter Plot - {features[0]}",
            width=800,
            height=600,
            xaxis_title="Concentration",
            yaxis_title="Uncertainty"
        )
        if show:
            du_plot.show()
            return None
        else:
            return du_plot

    def plot_feature_data(self, x_idx, y_idx, show: bool = True):
        """
        Create a plot of a data feature, column, vs another data feature, column. Specified by the feature indices.

        Parameters
        ----------
        x_idx : int
            The feature index for the x-axis values.
        y_idx: int
            The feature index for the y-axis values.
        """
        if x_idx > self.input_data.shape[1] - 1 or x_idx < 0:
            logger.info(f"Invalid x feature index provided, must be between 0 and {self.input_data.shape[1]}")
            return
        x_label = self.input_data.columns[x_idx]
        if y_idx > self.input_data.shape[1] - 1 or y_idx < 0:
            logger.info(f"Invalid y feature index provided, must be between 0 and {self.input_data.shape[1]}")
            return
        y_label = self.input_data.columns[y_idx]

        x_data = self.input_data[x_label]
        y_data = self.input_data[y_label]

        A = np.vstack([x_data.values, np.ones(len(x_data.values))]).T
        m, c = np.linalg.lstsq(A, y_data.values, rcond=None)[0]

        m1, c1 = np.linalg.lstsq(A, x_data.values, rcond=None)[0]

        xy_plot = go.Figure()
        xy_plot.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode='markers',
                name="Data",
                hovertemplate=(
                    "<b>Date:</b> %{customdata[0]}<br>"
                    f"<b>{x_label}:</b>" + " %{x}<br>"
                    f"<b>{y_label}:</b>" + " %{y}<extra></extra>"
                ),
                customdata=np.array([x_data.index]).T  # Pass index as custom data
            )
        )
        xy_plot.add_trace(
            go.Scatter(
                x=x_data,
                y=(m * x_data.values + c),
                line=dict(color='red', dash='dash', width=1),
                name='Regression'
            )
        )
        xy_plot.add_trace(
            go.Scatter(
                x=x_data,
                y=(m1 * x_data.values + c1),
                line=dict(color='blue', width=1),
                name='One-to-One'
            )
        )
        xy_plot.update_layout(
            title=f"Feature vs Feature Plot: {y_label}/{x_label}",
            width=800,
            height=600,
            xaxis_title=f"{x_label}",
            yaxis_title=f"{y_label}",
        )
        xy_plot.update_xaxes(range=[0, x_data.max() + 0.5])
        xy_plot.update_yaxes(range=[0, y_data.max() + 0.5])
        if show:
            xy_plot.show()
            return None
        else:
            return xy_plot

    def plot_feature_timeseries(self, feature_selection, show: bool = True):
        """
        Create a plot of a feature, or list of features, as a timeseries.

        Parameters
        ----------
        feature_selection : int or list
            A single or list of feature indices to plot as a timeseries.

        """
        if type(feature_selection) is int:
            feature_selection = feature_selection % self.input_data.shape[0]
            feature_selection = self.input_data.columns[feature_selection]
            feature_label = [feature_selection]
        else:
            if type(feature_selection) is list:
                feature_label = self.input_data.columns[feature_selection]
            else:
                feature_label = [feature_selection]
        data_df = copy.copy(self.input_data)
        data_df.index = pd.to_datetime(data_df.index)
        data_df = data_df.sort_index()
        data_df = data_df.resample(min_timestep(data_df)).mean()
        x = list(data_df.index)
        ts_plot = go.Figure()
        for feature_i in feature_label:
            y0 = data_df[feature_i]
            y = y0[x]
            ts_plot.add_trace(go.Scatter(x=x, y=y, line=dict(width=1), mode='lines+markers', name=feature_i))
        ts_plot.update_layout(title=f"Concentration Timeseries", width=800, height=600, hovermode='x unified')
        if len(feature_label) == 1:
            ts_plot.update_layout(showlegend=True)
        if show:
            ts_plot.show()
            return None
        else:
            return ts_plot

    @staticmethod
    def load_dataframe(input_df: pd.DataFrame, uncertainty_df: pd.DataFrame):
        """
        Pass in pandas dataframes for the input and uncertainty datasets, instead of using files.

        Parameters
        ----------
        input_df
        uncertainty_df

        Returns
        -------
        DataHandler
            Instance of DataHandler using dataframes as input.
        """
        dh = DataHandler(input_path="", uncertainty_path="", load=False)
        dh.input_data = input_df
        dh.uncertainty_data = uncertainty_df
        dh.features = input_df.columns
        dh._load_data(existing_data=True)
        dh._determine_optimal_block()
        return dh
