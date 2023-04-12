import numpy as np
from scipy import sparse
import numpy.linalg as nla


def _compute_update(H, PWW, PWV, penalty, alpha, regn, regd):
    violation = 0
    n_components = H.shape[0]

    for i in range(H.shape[1]):
        for k in range(n_components):

            if penalty == 2:
                regn = H[k, i] * alpha

            # gradient
            # g = GH[k, i] where GH = np.dot(PWW, H) - PWV
            g = - PWV[k, i] + regn

            for j in range(H.shape[0]):
                g += PWW[k, j] * H[j, i]

            # projected gradient
            pg = min(0, g) if H[k, i] == 0 else g

            # Hessian
            h = PWW[k, k] + regd

            # Update
            H[k, i] = max(H[k, i] - g / h, 0)

            violation += abs(pg)

    return violation


def safe_sparse_dot(a, b, *, dense_output=False):
    """Dot product that handle the sparse matrix case correctly.
    Parameters
    ----------
    a : {ndarray, sparse matrix}
    b : {ndarray, sparse matrix}
    dense_output : bool, default=False
        When False, ``a`` and ``b`` both being sparse will yield sparse output.
        When True, output will always be a dense array.
    Returns
    -------
    dot_product : {ndarray, sparse matrix}
        Sparse if ``a`` and ``b`` are sparse and ``dense_output=False``.
    """
    if a.ndim > 2 or b.ndim > 2:
        if sparse.issparse(a):
            # sparse is always 2D. Implies b is 3D+
            # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            ret = a @ b_2d
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sparse.issparse(b):
            # sparse is always 2D. Implies a is 3D+
            # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
            a_2d = a.reshape(-1, a.shape[-1])
            ret = a_2d @ b
            ret = ret.reshape(*a.shape[:-1], b.shape[1])
        else:
            ret = np.dot(a, b)
    else:
        ret = a @ b

    if (
        sparse.issparse(a)
        and sparse.issparse(b)
        and dense_output
        and hasattr(ret, "toarray")
    ):
        return ret.toarray()
    return ret


def nonzeros(m, row):
    """returns the non zeroes of a row in csr_matrix"""
    for index in range(m.indptr[row], m.indptr[row + 1]):
        yield m.indices[index], m.data[index]


def calculate_Q(residuals, uncertainty):
    return np.sum(np.sum(np.square(np.divide(residuals, uncertainty))))


def q_loss(V, U, W, H, uncertainty: bool = True):
    _wh = np.matmul(W, H)
    residuals = np.subtract(V, _wh)
    if uncertainty:
        residuals_u = np.divide(residuals, U)
        r2 = np.multiply(residuals_u, residuals_u)
        _q = np.sum(r2)
    else:
        _q = np.sum(np.multiply(residuals, residuals))
    return _q


def convert_matrix_order(matrix):
    new_matrix = matrix.flatten(order="A").reshape(matrix.shape)
    return new_matrix