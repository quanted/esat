import sys
import os
import re

module_path = os.path.abspath(os.path.join('..', "nmf_py"))
sys.path.append(module_path)

from src.model.nmf import NMF
from src.data.datahandler import DataHandler
from src.utils import q_loss, qr_loss, np_encoder
from tqdm import trange
from datetime import datetime
from scipy import optimize
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import logging
import copy
import pickle
import json
import time


logger = logging.getLogger("NMF")
logger.setLevel(logging.INFO)


class Constraint:
    """

    """

    type_list = ("pull down", "pull up", "pull to value", "set to zero", "set to base value", "define limits")

    def __init__(self,
                 constraint_type: str,
                 index: tuple,
                 target: str,
                 target_values=None
                 ):
        """
        Defines a constraint targeting a single factor element in the solution. There are 6 different constraint types:
        "pull down", "pull up", "pull to value", "set to zero", "set to base value", "define limits". A constraint must
        target a factor element in either the 'sample' or 'feature' solutions, which determines which matrix the
        provided index is targeting. The target_values of the constraint can take several forms depending on what
        constraint type is provided. Here are examples of each type and their target_values:

        Constraint(constraint_type='pull down', index=(0,0), target='feature', target_values=dQ_limit)
        Constraint(constraint_type='pull up', index=(0,0), target='feature', target_values=dQ_limit)
        Constraint(constraint_type='pull to value', index=(0,0), target='feature', target_values=(pull_to_value, dQ_limit))
        Constraint(constraint_type='set to zero', index=(0,0), target='feature')
        Constraint(constraint_type='set to base value', index=(0,0), target='feature')
        Constraint(constraint_type='define limits', index=(0,0), target='feature', target_values=(min_value, max_value))

        Parameters
        ----------
        constraint_type
        index
        target
        target_values
        """
        self.constraint_type = constraint_type
        self.index = index
        self.target = target
        self.target_values = target_values

    def describe(self, label):
        print_statement = f"Label: {label}, Type: {self.constraint_type}, Index: {self.index}, Target: {self.target}"
        if self.constraint_type in ('pull down', 'pull up'):
            print_statement += f", dQ: {self.target_values}"
        elif self.constraint_type == 'pull to value':
            print_statement += f", Value: {self.target_values[0]}, dQ: {self.target_values[1]}"
        elif self.constraint_type == 'define limits':
            print_statement += f", Min Value: {self.target_values[0]}, Max Value: {self.target_values[1]}"
        print(print_statement)


class ConstrainedModel:
    """

    """

    def __init__(self,
                 base_model: NMF,
                 data_handler: DataHandler,
                 softness: float = 1.0,
                 ):
        """

        Parameters
        ----------
        base_model
        softness
        """
        self.base_model = base_model
        self.dh = data_handler

        self.softness = softness
        self.soft_W = np.ones(shape=self.base_model.W.shape[0]) * self.softness
        self.soft_H = np.ones(shape=self.base_model.H.shape[0]) * self.softness
        self.soft_W2 = np.square(self.soft_W)
        self.soft_H2 = np.square(self.soft_H)

        self.constraints = {}
        self.expressions = []
        self.expression_labeled = None
        self.expression_mapped = None

        self.constrained_model = None
        self.Qaux = None

        self.metadata = {}

        self.update = self.base_model.update_step

    def add_constraint(self, constraint_type, index, target, target_values=None):
        """

        Parameters
        ----------
        constraint_type
        index
        target
        target_values

        Returns
        -------

        """
        constraint_label = ""
        if target == 'feature':
            if index[0] >= self.base_model.factors or index[0] < 0:
                logger.error(f"Error: Invalid constraint index. Provided: {index}. Valid ranges are "
                             f"(0:{self.base_model.factors-1}, 0:{self.base_model.n})")
                return False
            if index[1] >= self.base_model.n or index[1] < 0:
                logger.error(f"Error: Invalid constraint index. Provided: {index}. Valid ranges are "
                             f"(0:{self.base_model.factors-1}, 0:{self.base_model.n})")
                return False
            constraint_label = f"factor:{index[0]}|feature:{index[1]}"
        elif target == 'sample':
            if index[0] >= self.base_model.m or index[0] < 0:
                logger.error(f"Error: Invalid constraint index. Provided: {index}. Valid ranges are "
                             f"(0:{self.base_model.m-1}, 0:{self.base_model.factors})")
                return False
            if index[1] >= self.base_model.m or index[1] < 0:
                logger.error(f"Error: Invalid constraint index. Provided: {index}. Valid ranges are "
                             f"(0:{self.base_model.m-1}, 0:{self.base_model.factors})")
                return False
            constraint_label = f"factor:{index[0]}|sample:{index[1]}"
        if constraint_label in self.constraints.keys():
            logger.error(f"Error: Unable to set more than one constraint on a factor element. Existing constraint found.")
            return False
        new_constraint = Constraint(
            constraint_type=constraint_type,
            index=index,
            target=target,
            target_values=target_values
        )
        self.constraints[constraint_label] = new_constraint

    def remove_constraint(self, constraint_label):
        """

        Parameters
        ----------
        constraint_label

        Returns
        -------

        """
        self.constraints.pop(constraint_label)

    def list_constraints(self):
        """

        Returns
        -------

        """
        print(f"Constraint List - Count: {len(self.constraints.keys())}")
        for c_key, constraint in self.constraints.items():
            constraint.describe(label=c_key)

    def add_expression(self, expression):
        """

        Parameters
        ----------
        expression

        Returns
        -------

        """
        #TODO: Add validation of expressions to make sure they are properly structured.
        self.expressions.append(expression)

    def list_expressions(self):
        """

        Returns
        -------

        """
        for i,exp in enumerate(self.expressions):
            print(f"{i}: {exp}")

    def remove_expression(self, expression_idx):
        """

        Parameters
        ----------
        expression_idx

        Returns
        -------

        """
        if 0 <= expression_idx < len(self.expressions):
            del self.expressions[expression_idx]

    def _apply_constraints(self, iW, iH, D_w, D_h):
        for label, constraint in self.constraints.items():
            new_value = 0.0
            if constraint.target == "feature":
                current_value = iH[constraint.index]
            else:
                current_value = iW[constraint.index]

            if constraint.constraint_type == 'pull down':
                new_value = self._pull_down_value(iW=iW, iH=iH, target=constraint.target, index=constraint.index,
                                                  dQ_limit=constraint.target_values) - current_value
            elif constraint.constraint_type == 'pull up':
                new_value = self._pull_up_value(iW=iW, iH=iH, target=constraint.target, index=constraint.index,
                                                dQ_limit=constraint.target_values) - current_value
            elif constraint.constraint_type == 'pull to value':
                new_value = self._pull_to_value(iW=iW, iH=iH, target=constraint.target, index=constraint.index,
                                                target_value=constraint.target_values[0],
                                                dQ_limit=constraint.target_values[1]) - current_value
            elif constraint.constraint_type == 'set to zero':
                new_value = -1 * current_value
            elif constraint.constraint_type == 'set to base value':
                if constraint.target == "feature":
                    new_value = self.base_model.H[constraint.index] - current_value
                else:
                    new_value = self.base_model.W[constraint.index] - current_value
            elif constraint.constraint_type == 'define limits':
                if current_value < constraint.target_values[0]:
                    new_value = constraint.target_values[0] - current_value
                elif current_value > constraint.target_values[1]:
                    new_value = constraint.target_values[1] - current_value
                else:
                    new_value = current_value
            if constraint.target == "feature":
                D_h[constraint.index] = new_value
            else:
                D_w[constraint.index] = new_value
        return D_w, D_h

    def _apply_expressions(self, iW, iH, D_w, D_h):
        # Set the expression matrix and expression total vector to zero
        exp_A = np.zeros(shape=(len(self.expression_labeled), len(self.expression_mapped.keys())))
        exp_B = np.zeros(shape=len(self.expression_labeled))

        # If there are expressions.
        if len(self.expression_labeled) > 0:
            min_value = np.inf
            total_dQ = 0

            # Set the values of the expression matrix, from the coefficients in the expressions.
            # Set the expression total vector to the sum of the expression values from the factor elements in the expression
            for i, exp in enumerate(self.expression_labeled):
                total_dQ += exp[1]
                exp_b = 0
                for ele in exp[0]:
                    exp_details = self.expression_mapped[ele]
                    if exp_details['type'] == 'feature':
                        e_value = iH[exp_details['index']]
                    else:
                        e_value = iW[(exp_details['index'][1], exp_details['index'][0])]
                    exp_b += e_value
                    if e_value < min_value:
                        min_value = e_value
                    exp_index = exp_details['exp_i'].index(i)
                    coef_index = exp_details['coef_index']
                    coef_value = float(exp_details['coef'][exp_index])
                    exp_A[i, coef_index] = coef_value
                exp_B[i] = exp_b

            # Solve the expression matrices using a least squares solver from scipy.optimize (with solutions bounded between 0.0 and the max value of the total vector)
            exp_bounds = (0.0, np.max(exp_B))
            exp_X = optimize.lsq_linear(exp_A, exp_B, bounds=exp_bounds)

            # Assign the expression matrix solution values to the target matrix of H or W (depending on the factor element).
            for lbl, exp in self.expression_mapped.items():
                coef_index = exp['coef_index']
                ele_value = exp_X.x[coef_index]
                ele_index = exp['index']
                if exp['type'] == 'feature':
                    D_h[ele_index] = ele_value
                else:
                    D_w[ele_index[1], ele_index[0]] = ele_value

            # Convert the target matrices into difference matrices, where the values are the difference between the current solution and the target solution.
            _D_w = np.where(D_w != 0.0, D_w - iW, 0.0)
            _D_h = np.where(D_h != 0.0, D_h - iH, 0.0)

            # Constrain the target values to the W/H matrices to the total dQ allowed. If dQ is too large decrease the values by 10%, until dQ is lower then the dQ limit or max tries have been reached (in which case the target matrices are reset to zero).
            dQ_search = True
            max_search = 100
            while dQ_search:
                _W = iW + _D_w
                _H = iH + _D_h
                _q = q_loss(V=self.base_model.V, U=self.base_model.U, W=_W, H=_H)
                _qa = self._Qaux_loss(W=self.base_model.W, Wp=_W, Dw=_D_w, H=self.base_model.H, Hp=_H, Dh=_D_h)
                _dQ = (_q + _qa) - self.base_model.Qtrue
                if _dQ < total_dQ:
                    dQ_search = False
                else:
                    _D_w = _D_w * 0.9
                    _D_h = _D_h * 0.9
                if max_search < 0:
                    dQ_search = False
                    _D_w = np.zeros(shape=iW.shape)
                    _D_h = np.zeros(shape=iH.shape)
                max_search -= 1
            return _D_w, _D_h

    def _calculate_pull_matrices(self, iW, iH):
        # Set the target matrix values to zero
        D_w = np.zeros(shape=iW.shape)
        D_h = np.zeros(shape=iH.shape)

        D_w, D_h = self._apply_expressions(iW=iW, iH=iH, D_w=D_w, D_h=D_h)
        D_w, D_h = self._apply_constraints(iW=iW, iH=iH, D_w=D_w, D_h=D_h)
        return D_w, D_h

    def train(self, max_iterations: int = None, converge_delta: float = None, converge_n: int = None):
        """

        Parameters
        ----------
        max_iterations
        converge_delta
        converge_n

        Returns
        -------

        """
        max_iterations = self.base_model.metadata["max_iterations"] if max_iterations is None else max_iterations
        converge_delta = self.base_model.metadata["converge_delta"] if converge_delta is None else converge_delta
        converge_n = self.base_model.metadata["converge_n"] if converge_n is None else converge_n

        self._map_expressions()

        V = self.base_model.V
        U = self.base_model.U
        We = self.base_model.We
        W_i = copy.copy(self.base_model.W)
        H_i = copy.copy(self.base_model.H)

        t_iter = trange(max_iterations, desc="Q(Robust): NA, Q(Main): NA, Q(aux): NA", position=0, leave=True)
        qm_list = []
        Q_list = [[], [], []]
        i = 0
        Qtrue_i = None
        Qrobust_i = None
        Qaux_i = None

        converged = False
        for i in t_iter:
            D_w, D_h = self._calculate_pull_matrices(iW=W_i, iH=H_i)
            W_d = D_w + W_i
            H_d = np.abs(D_h + H_i)

            W_i, H_i = self.update(V=V, We=We, W=W_d, H=H_d)

            Qtrue_i = q_loss(V=V, U=U, W=W_i, H=H_i)
            Qrobust_i, _ = qr_loss(V=V, U=U, W=W_i, H=H_i)
            Qaux_i = self._Qaux_loss(W=self.base_model.W, Wp=W_i, Dw=D_w, H=self.base_model.H, Hp=H_i, Dh=D_h)
            Qmain_i = Qtrue_i + Qaux_i

            t_iter.set_description(f"Q(Robust): {round(Qrobust_i, 3)}, "
                                   f"Q(Main): {round(Qmain_i, 3)}, Q(aux): {round(Qaux_i, 3)}")
            Q_list[0].append(Qtrue_i)
            Q_list[1].append(Qrobust_i)
            Q_list[2].append(Qaux_i)

            qm_list.append(Qmain_i)
            if len(qm_list) > converge_n:
                qm_list.pop(0)
                if np.max(qm_list) - np.min(qm_list) <= converge_delta:
                    converged = True
                    break
        self.constrained_model = NMF(V=V, U=U, factors=self.base_model.factors, method=self.base_model.method,
                                     seed=self.base_model.seed)
        self.constrained_model.initialize(H=H_i, W=W_i)
        self.constrained_model.metadata["max_iterations"] = max_iterations
        self.constrained_model.metadata["converge_delta"] = converge_delta
        self.constrained_model.metadata["converge_n"] = converge_n
        self.constrained_model.Qrobust = Qrobust_i
        self.constrained_model.Qtrue = Qtrue_i
        self.constrained_model.converged = converged
        self.constrained_model.converge_steps = i

        self.Q_list = Q_list
        self.Qaux = Qaux_i
        logger.info(f"dQ(Robust): {round(Qrobust_i - self.base_model.Qrobust, 2)}, "
                    f"Q(Robust): {round(Qrobust_i, 2)}, "
                    f"% dQ(Robust): {round(100 * (Qrobust_i - self.base_model.Qrobust) / self.base_model.Qrobust, 2)}, "
                    f"Q(Aux): {round(Qaux_i, 2)}, Q(True): {round(Qtrue_i, 2)}, Converged: {converged}")

    def _map_expressions(self):
        exp_mapping = {}
        expressions_labeled = []
        terms = 0
        for exp_i in range(len(self.expressions)):
            expression_full = self.expressions[exp_i].split("=")
            expression_eq = expression_full[1].split(",")
            dQ = float(expression_eq[1])
            expression = re.split(r"[()]+", expression_full[0])[1:-1]
            neg_term = False
            expression_labeled = []
            for e in expression:
                if len(e) > 1:
                    e_terms0 = re.findall('\[(.*?)\]', e)[0]
                    e_terms1 = re.split(r"[:|]+", e_terms0)
                    e_index = (int(e_terms1[1]), int(e_terms1[3]))
                    e_coef = float(e.split("*")[0]) * (-1.0 if neg_term else 1.0)
                    if e_terms0 not in exp_mapping.keys():
                        exp_mapping[e_terms0] = {"coef": [e_coef], "exp_i": [exp_i], "index": e_index, "type": e_terms1[2], "coef_index": terms}
                        terms += 1
                    else:
                        exp_mapping[e_terms0]["coef"].append(e_coef)
                        exp_mapping[e_terms0]["exp_i"].append(exp_i)
                    expression_labeled.append(e_terms0)
                if e == "-":
                    neg_term = True
                else:
                    neg_term = False
            expressions_labeled.append((expression_labeled, dQ))
        self.expression_labeled = expressions_labeled
        self.expression_mapped = exp_mapping

    def _pull_down_value(self, iW, iH, target, index, dQ_limit):
        _H = copy.copy(iH)
        _W = copy.copy(iW)
        iQ = q_loss(V=self.base_model.V, U=self.base_model.U, W=_W, H=_H)

        high_mod = 1.0
        low_mod = 0.0
        modifier = (high_mod + low_mod) / 2

        value_found = False
        search_i = 0
        p_mod = 0.0
        new_value = 0.0
        max_seach_i = 30

        while not value_found:
            if target == "feature":
                new_value = _H[index] * modifier
                _H[index] = new_value
            else:
                new_value = _W[index] * modifier
                _W[index] = new_value
            _Q = q_loss(V=self.base_model.V, U=self.base_model.U, W=_W, H=_H)
            dQ = np.abs(iQ - _Q)
            if dQ > dQ_limit:
                low_mod = modifier
                modifier = (modifier + high_mod) / 2
            elif dQ < dQ_limit - 1e2:
                high_mod = modifier
                modifier = (low_mod + modifier) / 2
            else:
                value_found = True
            if np.abs(p_mod - modifier) <= 1e-8:  # small value, considered zero. Or no change in the modifier.
                value_found = True
            if search_i >= max_seach_i:
                value_found = True
            search_i += 1
            p_mod = modifier
        return new_value

    def _pull_up_value(self, iW, iH, target, index, dQ_limit):
        _H = copy.copy(iH)
        _W = copy.copy(iW)
        iQ = q_loss(V=self.base_model.V, U=self.base_model.U, W=iW, H=iH)

        high_modifier = 2.0
        low_modifier = 0.0
        high_found = False
        high_search_i = 0
        max_search_i = 30
        base_value = _H[index] if target == "feature" else _W[index]
        while not high_found:
            if target == "feature":
                new_value = base_value * high_modifier
                _H[index] = new_value
            else:
                new_value = base_value * high_modifier
                _W[index] = new_value
            _Q = q_loss(V=self.base_model.V, U=self.base_model.U, W=_W, H=_H)
            dQ = np.abs(iQ - _Q)
            if dQ < dQ_limit:
                low_modifier = high_modifier
                high_modifier *= 2
            else:
                high_found = True
            high_search_i += 1
            if high_search_i >= max_search_i:
                return new_value
        new_value = 0.0
        modifier = (high_modifier + low_modifier) / 2.0
        value_found = False
        search_i = 0
        while not value_found:
            if target == "feature":
                new_value = base_value * modifier
                _H[index] = new_value
            else:
                new_value = base_value * modifier
                _W[index] = new_value
            _Q = q_loss(V=self.base_model.V, U=self.base_model.U, W=_W, H=_H)
            dQ = np.abs(iQ - _Q)
            if dQ > dQ_limit:
                high_modifier = modifier
                modifier = (modifier + low_modifier) / 2.0
            elif dQ < dQ_limit - 1e2:
                low_modifier = modifier
                modifier = (high_modifier + modifier) / 2.0
            else:
                value_found = True
            search_i += 1
            if search_i >= max_search_i:
                value_found = True
        return new_value

    def _pull_to_value(self, iW, iH, target, index, target_value, dQ_limit):
        _H = copy.copy(iH)
        _W = copy.copy(iW)
        iQ = q_loss(V=self.base_model.V, U=self.base_model.U, W=_W, H=_H)

        # Check if target value is within dQ_limit, if true return.
        if target == "feature":
            current_value = _H[index]
            _H[index] = target_value
        else:
            current_value = _W[index]
            _W[index] = target_value
        iQ2 = q_loss(V=self.base_model.V, U=self.base_model.U, W=_W, H=_H)
        if np.abs(iQ - iQ2) < dQ_limit:
            return target_value

        if target_value > current_value:
            high_value = target_value
            low_value = current_value
        else:
            high_value = current_value
            low_value = target_value

        mid_value = (high_value + low_value) / 2

        value_found = False
        search_i = 0
        p_value = 0.0
        new_value = 0.0
        max_search_i = 30

        while not value_found:
            if target == "feature":
                new_value = mid_value
                _H[index] = new_value
            else:
                new_value = mid_value
                _W[index] = new_value
            _Q = q_loss(V=self.base_model.V, U=self.base_model.U, W=_W, H=_H)
            dQ = np.abs(iQ - _Q)
            if dQ > dQ_limit:
                low_value = mid_value
                mid_value = (mid_value + high_value) / 2
            elif dQ < dQ_limit - 1e2:
                high_value = mid_value
                mid_value = (low_value + mid_value) / 2
            else:
                value_found = True
            if np.abs(p_value - mid_value) <= 1e-8:  # small value, considered zero. Or no change in the modifier.
                value_found = True
            if search_i >= max_search_i:
                value_found = True
            search_i += 1
            p_value = mid_value
        return new_value

    def _Qaux_loss(self, W, Wp, Dw, H, Hp, Dh):
        w_r = np.square(W + Dw - Wp)
        w_qaux = np.divide(w_r.sum(axis=1), self.soft_W2)
        h_r = np.square(H + Dh - Hp)
        h_qaux = np.divide(h_r.sum(axis=1), self.soft_H2)
        return np.sum(w_qaux) + np.sum(h_qaux)

    def display_results(self):
        """

        Returns
        -------

        """
        logger.info(f"dQ(Robust): {round(self.constrained_model.Qrobust - self.base_model.Qrobust, 2)}, "
                    f"Q(Robust): {round(self.constrained_model.Qrobust, 2)}, "
                    f"Base Q(Robust): {round(self.base_model.Qrobust, 2)}, "
                    f"% dQ(Robust): {round(100 * ((self.constrained_model.Qrobust - self.base_model.Qrobust) / self.base_model.Qrobust), 2)}, "
                    f"Q(Aux): {round(self.Qaux, 2)}, Q(True): {round(self.constrained_model.Qtrue, 2)}, "
                    f"Base Q(True): {round(self.base_model.Qtrue, 2)}, "
                    f"Converged: {self.constrained_model.converged}")

    def plot_Q(self, Qtype: str = 'True'):
        """

        Parameters
        ----------
        Qtype

        Returns
        -------

        """
        title = 'Q(True)'
        q_index = 0
        if Qtype.lower() == "robust":
            q_index = 1
            title = 'Q(Robust)'
        elif Qtype.lower() == "aux":
            q_index = 2
            title = 'Q(aux)'

        x = list(range(len(self.Q_list[q_index])))
        y = self.Q_list[q_index]
        q_plot = go.Figure()
        q_plot.add_trace(go.Scatter(x=x, y=y))
        q_plot.update_layout(height=800, width=800, title=f"{title}")
        q_plot.update_yaxes(title_text="Q")
        q_plot.update_xaxes(title_text="Iterations")
        q_plot.show()

    def plot_profile_contributions(self, factor_idx):
        """

        Parameters
        ----------
        factor_idx

        Returns
        -------

        """
        if factor_idx is not None:
            if factor_idx > self.constrained_model.factors or factor_idx < 1:
                logger.warn(f"Invalid factor_idx provided, must be between 1 and {self.constrained_model.factors}")
                return
        self.plot_profile(factor_idx=factor_idx)
        self.plot_contributions(factor_idx=factor_idx)

    def plot_profile(self, factor_idx):
        """

        Parameters
        ----------
        factor_idx

        Returns
        -------

        """
        factor_label = factor_idx
        factor_idx = factor_idx - 1

        b_W = self.base_model.W[:, factor_idx]
        b_H = self.base_model.H[factor_idx]
        b_H_sum = self.base_model.H.sum(axis=0)
        b_factor_matrix = np.matmul(b_W.reshape(len(b_W), 1), [b_H])

        b_factor_conc_sum = b_factor_matrix.sum(axis=0)
        b_factor_conc_sum[b_factor_conc_sum == 0.0] = 1e-12

        i_W = self.constrained_model.W[:, factor_idx]
        i_H = self.constrained_model.H[factor_idx]
        i_H_sum = self.constrained_model.H.sum(axis=0)
        i_factor_matrix = np.matmul(i_W.reshape(len(i_W), 1), [i_H])

        i_factor_conc_sum = i_factor_matrix.sum(axis=0)
        i_factor_conc_sum[i_factor_conc_sum == 0] = 1e-12

        b_norm_H = np.round(100 * (b_H / b_H_sum), 2)
        i_norm_H = np.round(100 * (i_H / i_H_sum), 2)

        # Highlight the factor feature elements which were in an expression.
        marker_line_colors = ['rgb(125,220,157)'] * len(self.dh.features)
        for factor_element, fe_details in self.expression_mapped.items():
            if 'feature' in factor_element:
                factor_j = fe_details['index'][0]
                factor_feature_k = fe_details['index'][1]
                if factor_j == factor_idx:
                    marker_line_colors[factor_feature_k] = 'rgb(255,44,44)'
        # Highlight the factor feature elements which were constrained
        for c_key, constraint in self.constraints.items():
            if constraint.constraint_type == "feature" and constraint.index[0] == factor_idx:
                marker_line_colors[constraint.index[1]] = 'rgb(122,44,255)'

        fig = make_subplots(specs=[[{"secondary_y": True}]], rows=1, cols=1)
        fig.add_trace(go.Scatter(x=self.dh.features, y=b_norm_H, mode="markers", marker=dict(color='gray'),
                                 name="Base % of Features", opacity=0.8), secondary_y=True, row=1, col=1)
        fig.add_trace(go.Scatter(x=self.dh.features, y=i_norm_H, mode="markers", marker=dict(color='red'),
                                 name="Constrained % of Features", opacity=0.6), secondary_y=True, row=1, col=1)
        fig.add_trace(go.Bar(x=self.dh.features, y=b_factor_conc_sum, marker_color='rgb(203,203,203)',
                             marker_line_color='rgb(186,186,186)', marker_line_width=1.5, opacity=0.6,
                             name='Base Conc. of Features'), secondary_y=False, row=1, col=1)
        fig.add_trace(go.Bar(x=self.dh.features, y=i_factor_conc_sum, marker_color='rgb(134,236,168)',
                             marker_line_color='rgb(125,220,157)', marker_line_width=1.5, opacity=0.6,
                             name='Constrained Conc. of Features'), secondary_y=False, row=1, col=1)
        fig.update_layout(width=1200, height=600, title=f"Constrained Factor Profile - Factor {factor_label}",
                          barmode='group', scattermode='group', hovermode="x unified")
        fig.update_yaxes(type="log", secondary_y=False, range=[0, np.log10(b_factor_conc_sum).max()], row=1, col=1)
        fig.update_yaxes(secondary_y=True, range=[0, 100])
        fig.show()

    def plot_contributions(self, factor_idx):
        """

        Parameters
        ----------
        factor_idx

        Returns
        -------

        """
        factor_label = f"Factor {factor_idx}"
        factor_idx = factor_idx - 1

        b_W = self.base_model.W[:, factor_idx]
        i_W = self.constrained_model.W[:, factor_idx]

        b_norm_contr = b_W / b_W.mean()
        b_data_df = copy.copy(self.dh.input_data)
        b_data_df[factor_label] = b_norm_contr
        b_data_df.index = pd.to_datetime(b_data_df.index)
        b_data_df = b_data_df.sort_index()
        b_data_df = b_data_df.resample('D').mean()

        i_norm_contr = i_W / i_W.mean()
        i_data_df = copy.copy(self.dh.input_data)
        i_data_df[factor_label] = i_norm_contr
        i_data_df.index = pd.to_datetime(i_data_df.index)
        i_data_df = i_data_df.sort_index()
        i_data_df = i_data_df.resample('D').mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=b_data_df.index, y=b_data_df[factor_label], mode='lines+markers',
                                 marker_color='rgb(186,186,186)', name="Base Factor Contributions"))
        fig.add_trace(go.Scatter(x=i_data_df.index, y=i_data_df[factor_label], mode='lines+markers',
                                 marker_color='rgb(125,220,157)', name="Constrained Factor Contributions"))
        fig.update_layout(width=1200, height=800,
                          title=f"Constrained Factor Contributions - Factor {factor_label}",
                          hovermode="x unified")
        fig.update_yaxes(title_text="Factor Contributions")
        fig.show()

    def plot_factor_fingerprints(self):
        """

        Parameters
        ----------

        Returns
        -------

        """
        b_H = self.base_model.H
        c_H = self.constrained_model.H

        b_normalized = 100 * (b_H / b_H.sum(axis=0))
        c_normalized = 100 * (c_H / c_H.sum(axis=0))

        fp_factors_fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       subplot_titles=("Base Profile", "Constrained Profile"), vertical_spacing=0.075)
        colors = px.colors.sequential.Viridis_r
        for idx in range(self.base_model.factors - 1, -1, -1):
            fp_factors_fig.add_trace(go.Bar(name=f"Base Factor {idx + 1}", x=self.dh.features, y=b_normalized[idx],
                                            marker_color=colors[idx]), row=1, col=1)
            fp_factors_fig.add_trace(
                go.Bar(name=f"Constrained Factor {idx + 1}", x=self.dh.features, y=c_normalized[idx],
                       marker_color=colors[idx]), row=2, col=1)
        fp_factors_fig.update_layout(title=f"Constrained Factor Fingerprints", width=1200, height=800,
                                     barmode='stack', hovermode='x unified')
        fp_factors_fig.update_yaxes(title_text="% Feature Concentration", range=[0, 100])
        fp_factors_fig.show()

    def plot_g_space(self, factor_idx1, factor_idx2, show_base: bool = False, show_delta: bool = False):
        """

        Parameters
        ----------
        factor_idx1
        factor_idx2
        show_base
        show_delta

        Returns
        -------

        """
        if factor_idx1 is not None:
            if factor_idx1 > self.base_model.factors or factor_idx1 < 1:
                logger.warn(f"Invalid factor_idx1 provided, must be between 1 and {self.base_model.factors}")
                return
        if factor_idx2 is not None:
            if factor_idx2 > self.base_model.factors or factor_idx2 < 1:
                logger.warn(f"Invalid factor_idx2 provided, must be between 1 and {self.base_model.factors}")
                return
        f1_idx = factor_idx1 - 1
        f2_idx = factor_idx2 - 1

        b_W = self.base_model.W
        c_W = self.constrained_model.W

        b_normalized_factors_contr = b_W / b_W.sum(axis=0)
        c_normalized_factors_contr = c_W / c_W.sum(axis=0)

        if show_delta:
            arrows = ((c_normalized_factors_contr[:, f1_idx] - b_normalized_factors_contr[:, f1_idx]),
                      (c_normalized_factors_contr[:, f2_idx] - b_normalized_factors_contr[:, f2_idx]))
            fp_g_fig = ff.create_quiver(x=b_normalized_factors_contr[:, f1_idx],
                                        y=b_normalized_factors_contr[:, f2_idx], u=arrows[0], v=arrows[1],
                                        name="Constrained Delta", line_width=1, arrow_scale=0.01, scale=0.99)
            fp_g_fig.add_trace(
                go.Scatter(x=c_normalized_factors_contr[:, f1_idx], y=c_normalized_factors_contr[:, f2_idx],
                           mode='markers', name="Constrained"))
            fp_g_fig.add_trace(
                go.Scatter(x=b_normalized_factors_contr[:, f1_idx], y=b_normalized_factors_contr[:, f2_idx],
                           mode='markers', name="Base"))
        else:
            fp_g_fig = go.Figure()
            fp_g_fig.add_trace(
                go.Scatter(x=c_normalized_factors_contr[:, f1_idx], y=c_normalized_factors_contr[:, f2_idx],
                           mode='markers', name="Fpeak"))
            if show_base:
                fp_g_fig.add_trace(
                    go.Scatter(x=b_normalized_factors_contr[:, f1_idx], y=b_normalized_factors_contr[:, f2_idx],
                               mode='markers', name="Base"))
        fp_g_fig.update_layout(title=f"Constrained G-Space Plot", width=800, height=800)
        fp_g_fig.update_yaxes(title_text=f"Factor {factor_idx1} Contributions (avg=1)")
        fp_g_fig.update_xaxes(title_text=f"Factor {factor_idx2} Contributions (avg=1)")
        fp_g_fig.show()

    def plot_factor_contributions(self, feature_idx, threshold: float = 0.06):
        """

        Parameters
        ----------
        feature_idx
        threshold

        Returns
        -------

        """
        if feature_idx is not None:
            if feature_idx < 1 or feature_idx > self.base_model.n:
                logger.warn(f"Invalid feature_idx provided, must be between 1 and {self.base_model.n}")
                return
            feature_idx = feature_idx - 1
        else:
            feature_idx = 0
        x_label = self.dh.input_data.columns[feature_idx]
        factors_data = self.constrained_model.H
        normalized_factors_data = 100 * (factors_data / factors_data.sum(axis=0))

        feature_contr = normalized_factors_data[:, feature_idx]
        feature_contr_inc = []
        feature_contr_labels = []
        feature_legend = {}
        for idx in range(feature_contr.shape[0] - 1, -1, -1):
            idx_l = idx + 1
            if feature_contr[idx] > threshold:
                feature_contr_inc.append(feature_contr[idx])
                feature_contr_labels.append(f"Factor {idx_l}")
                feature_legend[f"Factor {idx_l}"] = f"Factor {idx_l} = {factors_data[idx:, feature_idx]}"
        feature_fig = go.Figure(data=[
            go.Pie(labels=feature_contr_labels, values=feature_contr_inc, hoverinfo="label+value", textinfo="percent")])
        feature_fig.update_layout(title=f"Factor Contributions to Feature: {x_label}", width=1200,
                                  height=600,
                                  legend_title_text=f"Factor Contribution > {threshold}%")
        feature_fig.show()

        factors_contr = self.constrained_model.W
        normalized_factors_contr = 100 * (factors_contr / factors_contr.sum(axis=0))
        factor_labels = [f"Factor {i}" for i in range(1, normalized_factors_contr.shape[1] + 1)]
        contr_df = pd.DataFrame(normalized_factors_contr, columns=factor_labels)
        contr_df.index = pd.to_datetime(self.dh.input_data.index)
        contr_df = contr_df.sort_index()
        contr_df = contr_df.resample('D').mean()

        contr_fig = go.Figure()
        for factor in factor_labels:
            contr_fig.add_trace(go.Scatter(x=contr_df.index, y=contr_df[factor], mode='lines+markers', name=factor))
        contr_fig.update_layout(title=f"Factor Contributions (avg=1)",
                                width=1200, height=600,
                                legend=dict(orientation="h", xanchor="right", yanchor="bottom", x=1, y=1.02))
        contr_fig.update_yaxes(title_text="Normalized Contribution")
        contr_fig.show()

    def save(self):
        pass

    @staticmethod
    def load():
        pass
