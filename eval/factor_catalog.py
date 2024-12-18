import logging
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm, tnrange
from esat_eval.factor_comparison import FactorCompare

logger = logging.getLogger(__name__)


class Factor:
    """
    A Factor instance is used to store and manage the factors that are generated among different independent models.

    Parameters
    ----------
    name : str
        The name of the factor.
    profile : np.ndarray
        The factor profile array.
    contribution : np.ndarray
        The factor contribution array.
    model_idx : int
        The index of the model that this factor was sourced from.
    method : str
        The correlation method to use, options include: 'raae', 'corr', 'emc'.
    threshold : float
        The correlation threshold to use for adding a new factor to the catalog. Defaults to 0.1 for RAAE and 0.85 for
        EMC and CORR
    """
    def __init__(self, name, profile, contribution, model_idx, method: str = "raae", threshold: float = None):
        self.name = name
        self.factor = [(profile, contribution)]       # Each factor consists of a tuple (H_k, W_k) the profile and the contribution
        self.model = [model_idx]
        self.correlation = [ 0.0 if method == "raae" else 1.0]
        self.samples = len(contribution)
        self.method = method
        if threshold is None:
            if method == "raae":
                self.threshold = 0.1
            else:
                self.threshold = 0.9

    def __repr__(self):
        return f"Factor: {self.name}, Count: {len(self.model)}, Mean Corr: {np.round(np.mean(self.correlation),4)}, Method: {self.method}"

    def check_matrix(self, wh_j):
        """
        Check the correlation between a given factor matrix and this factor instance matrices.

        Parameters
        ----------
        wh_j: np.ndarray
            The factor matrix to compare to the Factor instance.

        Returns
        -------
        float
            The mean correlation of the factor to the existing factors in the Factor instance.
        """
        corr_list = []
        for f in self.factor:
            f_H, f_W = f
            f_W = f_W.reshape(len(f_W), 1)
            f_wh = np.matmul(f_W, [f_H])
            corr = FactorCompare.calculate_correlation(factor1=f_wh.flatten(),
                                                      factor2=wh_j.flatten())
            corr_list.append(corr)
        mean_corr = np.mean(corr_list)
        return mean_corr


    def check(self, H_j, W_j, profile_only: bool = False):
        """
        Check the correlation between a given factor and this factor instance using one of the three correlation methods.

        Parameters
        ---------
        H_j : np.ndarray
            The factor profile array to compare to the Factor instance.
        W_j : np.ndarray
            The factor contribution array to compare to the Factor instance.
        profile_only : bool
            Only check the profile of the factor.

        Returns
        -------
        float
            The mean correlation of the factor to the existing factors in the Factor instance
        """
        n = 1 / len(W_j)
        j_W_mean = np.mean(W_j, axis=0)
        j_mass_matrix = (j_W_mean*H_j)/np.sum(j_W_mean*H_j)

        corr_list = []
        for f in self.factor:
            f_H, f_W = f
            if profile_only:
                corr = FactorCompare.calculate_correlation(factor1=f_H, factor2=H_j)
            elif self.method == "raae":
                corr = (np.sum(np.abs(f_W - W_j)) * n) / (np.sum(f_W) * n)
            elif self.method == "emc":
                f_W_mean = np.mean(f_W, axis=0)
                f_mass_matrix = (f_W_mean * f_H) / np.sum(f_W_mean * f_H)
                corr = FactorCompare.calculate_correlation(factor1=f_mass_matrix,
                                                          factor2=j_mass_matrix)
            else:
                # R2 between the contribution factors and the provided factor contribution
                corr = FactorCompare.calculate_correlation(factor1=f_W.flatten(),
                                                         factor2=W_j.flatten())
            corr_list.append(corr)
        mean_corr = np.mean(corr_list)
        return mean_corr

    def add(self, H_j, W_j, model_index, corr):
        """
        Add a new factor to the factor instance.

        Parameters
        ----------
        H_j : np.ndarray
            The factor profile array to add to the Factor instance.
        W_j : np.ndarray
            The factor contribution array to add to the Factor instance
        model_index : int
            The index of the model that this factor was sourced from.
        corr : float
            The correlation of the factor to the existing factors in the Factor instance.

        """
        self.factor.append((H_j, W_j))
        self.model.append(model_index)
        self.correlation.append(corr)


class FactorCatalog:
    """
    The Factor catalog is used to store, compare and manage a collection of different factors that have been detected
    among different independent models. When a factor is added to the catalog, a correlation comparison is made between
    that factor and all other factors present in the catalog. The correlation metric used can be specified as either:
    'raae', 'corr', or 'emc'. The correlation metrics are described in https://doi.org/10.1021/es800085t.

    corr: is defined as the R^2 correlation between the factor contribution vectors of two factors. In some comparisons, if
    only the factor profiles are provided, they are used for the correlation instead of the factor contributions.

    raae: is defined as the relative average absolute error between the factor contribution vectors of two factors.

    emc: is defined as the correlation between the mass matrix of the factor contribution vectors of two factors.

    A cataloged factor consists of all the factors that exceed the specified correlation threshold, so that when a new
    factor is added it must exceed the threshold with all individual factors in that cataloged factor.

    The catalog can be used to determine the frequency of occurrence of a given factor in the batch of models.

    Parameters
    ----------
    method : str
        The correlation method to use, options include: 'raae', 'corr', 'emc'.
    threshold : float
        The correlation threshold to use for adding a new factor to the catalog.
    exclude_outliers : bool
        Exclude outlier models from consideration when adding a new factor to the catalog. An outlier model is defined
        as a model whose loss value Q is greater than outlier threshold of the min loss value for the batch.
    outlier_threshold : float
        The threshold multiplier of the min loss for determining an outlier model.

    """
    def __init__(self,
                 method: str = "raae",
                 threshold: float = 0.9,
                 exclude_outliers: bool = False,
                 outlier_threshold: float = 1.5,
                 ):
        self.factors = {}       # A dictionary of factors, each factor is a Factor instance
        self.threshold = threshold
        self.exclude_outliers = exclude_outliers
        self.outlier_threshold = outlier_threshold
        self.method = method
        self.idx = 0
        self.model_count = 0

    def summary(self):
        """
        Print a summary of the factor catalog.
        """
        for i, f in self.factors.items():
            f_summary = repr(f) + f", %: {np.round(100*len(f.model)/self.model_count)}"
            print(f_summary)

    def to_df(self):
        """
        Convert the factor catalog instance to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the factor catalog.

        """
        factor_occurrence = []
        factor_count = []
        factor_mean_corr = []
        for _, f in self.factors.items():
            factor_occurrence.append(round(100 * len(f.model)/self.model_count, 2))
            factor_count.append(len(f.model))
            factor_mean_corr.append(round(np.mean(f.correlation), 4))
        factors_df = pd.DataFrame(data={"factor": list(self.factors.keys()), "count": factor_count,  "% occurrence": factor_occurrence, "mean corr": factor_mean_corr})
        factors_df = factors_df.sort_values(by='% occurrence', ascending=False)
        return factors_df

    def check_factor(self, H_i, W_i, profile_only: bool = False):
        """
        Check the correlation of a given factor to all factors in the catalog.

        Parameters
        ----------
        H_i : np.ndarray
            Factor profile vector
        W_i : np.ndarray
            Factor contribution vector
        profile_only : bool
            Only check the profile of the factor.

        Returns
        -------
        str
            The best factor in the catalog
        float
            The correlation of the best factor in the catalog
        """
        best_factor = None
        best_corr = 1.0 if self.method == "raae" else 0.0
        for f_label, f in self.factors.items():
            corr = f.check(H_i, W_i, profile_only)
            if self.method == "raae":
                if corr < best_corr:
                    best_corr = corr
                    best_factor = f_label
            else:
                if corr > best_corr:
                    best_corr = corr
                    best_factor = f_label
        return best_factor, best_corr

    def check_matrix(self, wh_i):
        """
        Check the correlation of a given factor matrix to all factors in the catalog.

        Parameters
        ----------
        wh_i : np.ndarray
            The factor matrix to compare to the catalog.

        Returns
        -------
        str
            The factor in the catalog with the highest correlation
        float
            The best correlation of the compared factors in the catalog
        """
        best_factor = None
        best_corr = 0.0
        for f_label in tqdm(self.factors.keys(), desc="Comparing Cataloged Factor Matrices", leave=True):
            f = self.factors[f_label]
            corr = f.check_matrix(wh_i)
            if corr > best_corr:
                best_corr = corr
                best_factor = f_label
        return best_factor, best_corr

    def add_factor(self, H_i, W_i, model_index, verbose: bool = False):
        """
        Add a new factor to the catalog.

        Parameters
        ----------
        H_i : np.ndarray
            The factor profile matrix.
        W_i : np.ndarray
            The factor contribution matrix.
        model_index : int
            The index of the model that this factor was sourced from.
        verbose : bool
            Display the results.

        """
        if self.model_count < model_index:
            self.model_count = model_index+1
        add_new = False
        if self.idx == 0:
            add_new = True
        best_factor, best_corr = self.check_factor(H_i=H_i, W_i=W_i)
        if self.method == "raae":
            if best_corr > self.threshold:
                add_new = True
        else:
            if best_corr < self.threshold:
                add_new = True

        if add_new:
            if verbose:
                logger.info(f"New Factor: {self.idx}, Model: {model_index}, Method: {self.method}, "
                            f"Best CORR: {best_corr}")
            new_factor = Factor(name=self.idx, profile=H_i, contribution=W_i, model_idx=model_index,
                                method=self.method, threshold=self.threshold)
            self.factors[self.idx] = new_factor
            self.idx += 1
        else:
            if verbose:
                logger.info(f"Adding to Factor: {best_factor}, Model: {model_index}, Method: {self.method}, "
                            f"Best CORR: {best_corr}")
            self.factors[best_factor].add(H_j=H_i, W_j=W_i, model_index=model_index, corr=best_corr)

    def add_batch(self, batch_results, verbose: bool = False):
        """
        Analyze a batch of models and add the factors to the catalog.

        Parameters
        ----------
        batch_results : BatchSA
            A completed instance of BatchSA, batch of models, to analyze.
        verbose : bool
            Display the results.

        """
        loss_min = None
        if self.exclude_outliers:
            loss_list = []
            for model in batch_results.results:
                loss_list.append(model.Qtrue)
            loss_min = np.min(loss_list)
            logger.info(f"Excluding models with loss greater than {np.round(loss_min * self.outlier_threshold, 4)}")

        for i in tnrange(len(batch_results.results), desc="Analyzing Models", leave=True):
            model = batch_results.results[i]
            if self.exclude_outliers:
                if model.Qtrue > loss_min * self.outlier_threshold:
                    continue
            for f in range(model.factors):
                H_f = model.H[f]
                W_f = model.W[:,f]
                self.add_factor(H_i=H_f, W_i=W_f, model_index=i, verbose=verbose)

    def compare_factor(self, factor):
        """
        Compare a factor to the cataloged factors.

        Parameters
        ----------
        factor : Factor
            The factor to compare to the cataloged Factor.

        Returns
        -------
        str
            Most correlated factor in the catalog
        float
            The correlation of the most correlated factor in the catalog
        """
        best_factor = None
        best_corr = 0.0
        for f_label in tqdm(self.factors.keys(), desc="Comparing Cataloged Factors", leave=True):
            f = self.factors[f_label]
            corr_list = []
            for fi in tqdm(factor.factor, desc="Comparing Factor Members", leave=False):
                corr = f.check(H_j=fi[0], W_j=fi[1], profile_only=len(fi[1]) != f.samples)
                corr_list.append(corr)
            mean_corr = np.mean(corr_list)
            if mean_corr > best_corr:
                best_corr = mean_corr
                best_factor = f_label
        return best_factor, best_corr
