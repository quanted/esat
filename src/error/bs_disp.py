import logging
import pickle
import os
import copy
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
from pathlib import Path

from src.model.nmf import NMF
from src.error.bootstrap import Bootstrap
from src.error.displacement import Displacement

logger = logging.getLogger("NMF")
logger.setLevel(logging.INFO)


class BSDISP:
    """
    The Bootstrap-Displacement (BS-DISP) method combines both the Bootstrap and Displacement methods to estimate the
    errors with both random and rotational ambiguity. For each BS run/dataset, the DISP method is run on that dataset.
    """

    dQmax = [4, 2, 1, 0.5]

    def __init__(self,
                 nmf: NMF,
                 feature_labels: list,
                 model_selected: int = -1,
                 bootstrap: Bootstrap = None,
                 bootstrap_n: int = 20,
                 block_size: int = 10,
                 threshold: float = 0.6,
                 max_search: int = 50,
                 threshold_dQ: float = 0.1,
                 seed: int = None,
                 ):
        """

        Parameters
        ----------
        nmf : NMF
           A completed NMF base model that used the same data and uncertainty datasets.
        feature_labels : list
           The labels for the features, columns of the dataset, specified from the data handler.
        model_selected : int
           The index of the model selected from a batch NMF run, used for labeling.
        bootstrap: Bootstrap
           A previously complete BS model.
        bootstrap_n : int
           The number of bootstrap runs to make.
        block_size : int
           The block size for the BS resampling.
        threshold : float
           The correlation threshold that must be met for a BS factor to be mapped to a base model factor, factor
           correlations must be greater than the threshold or are labeled unmapped.
        max_search : int
           The maximum number of search steps to complete when trying to find a factor feature value. Default = 50
        threshold_dQ : float
           The threshold range of the dQ value for the factor feature value to be considered found. I.E, dQ=4 and
           threshold_dQ=0.1, than any value between 3.9 and 4.0 will be considered valid.
        seed : int
           The random seed for random resampling of the BS datasets. The base model random seed is used for all BS runs,
           which result in the same initial W matrix.
        """
        self.nmf = nmf
        self.feature_labels = feature_labels
        self.model_selected = model_selected
        self.bootstrap = bootstrap
        self.bootstrap_n = bootstrap_n if bootstrap is None else bootstrap.bootstrap_n
        self.block_size = block_size if bootstrap is None else bootstrap.block_size
        self.threshold = threshold if bootstrap is None else bootstrap.threshold
        self.max_search = max_search
        self.threshold_dQ = threshold_dQ
        self.seed = seed if bootstrap is None else bootstrap.bs_seed
        self.disp_results = {}

    def run(self,
            keep_H: bool = True,
            reuse_seed: bool = True,
            block: bool = True,
            overlapping: bool = False):
        """
        Run the BS-DISP error estimation method. If no prior BS run had been completed, this will execute a BS run and
        then a DISP for each of the BS runs.

        Parameters
        ----------
        keep_H : bool
           When retraining the NMF models using the resampled input and uncertainty datasets, keep the base model H
           matrix instead of reinitializing. The W matrix is always reinitialized when NMF is run on the BS datasets.
           Default = True
        reuse_seed : bool
           Reuse the base model seed for initializing the W matrix, and the H matrix if keep_H = False. Default = True
        block : bool
           Use block resampling instead of full resampling. Default = True
        overlapping : bool
           Allow resampled blocks to overlap. Default = False

        Returns
        -------

        """
        if self.bootstrap is None:
            # Run BS
            self.bootstrap = Bootstrap(nmf=self.nmf, feature_labels=self.feature_labels,
                                       model_selected=self.model_selected, block_size=self.block_size,
                                       bootstrap_n=self.bootstrap_n, threshold=self.threshold, seed=self.seed)
            self.bootstrap.run(keep_H=keep_H, reuse_seed=reuse_seed, block=block, overlapping=overlapping)
        for i, bs_result in self.bootstrap.bs_results.items():
            bs_model = bs_result["model"]
            i_disp = Displacement(nmf=bs_model, feature_labels=self.feature_labels, selected_model=self.model_selected,
                                  threshold_dQ=self.threshold_dQ, max_search=self.max_search)
            i_disp.dQmax = self.dQmax
            i_disp.run()
            self.disp_results[i] = i_disp

    def __compile_results(self):
        """
        Calculate the merging statistics and metrics for the disp results.
        """
        pass

    def summary(self):
        """
        Prints a summary of the BS-DISP results table.

        Summary shows the largest change in Q across all DISP runs, the % of cases with a drop of Q, swap in best
        fit and swap in DISP phase. Followed by the swap % table as shown in the regular DISP summary. The dQmax values
        in BS-DISP differ from DISP to account for increased variability, BS-DISP dQmax values are (0.5, 1, 2, 4) while
        DISP dQmax values are (4, 8, 16, 32)
        """
        pass