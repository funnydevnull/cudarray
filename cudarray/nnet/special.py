import numpy as np
import cudarray as ca
from ..wrap import nnet


def softmax(x):
    e = ca.exp(x - ca.amax(x, axis=1, keepdims=True))
    return e/ca.sum(e, axis=1, keepdims=True)


def categorical_cross_entropy(y_pred, y_true, eps=1e-15):
    # Assumes one-hot encoding.
    y_pred = ca.clip(y_pred, eps, 1 - eps)
    # XXX: do we need to normalize?
    y_pred /= ca.sum(y_pred, axis=1, keepdims=True)
    loss = -ca.sum(y_true * ca.log(y_pred), axis=1)
    return loss


def one_hot_encode(labels, n_classes, out=None):
    out_shape = (labels.size, n_classes)
    if labels.dtype != np.dtype('int32'):
        raise ValueError('labels.dtype must be int')
    if out is None:
        out = ca.empty(out_shape)
    else:
        if out.shape != out_shape:
            raise ValueError('shape mismatch')
    nnet._one_hot_encode(labels._data, n_classes, out_shape[0], out._data)
    return out


def one_hot_decode(one_hot, out=None):
    out_shape = (one_hot.shape[0],)
    if out is None:
        out = ca.empty(out_shape, dtype=np.dtype('int32'))
    else:
        if out.dtype != np.dtype('int32'):
            raise ValueError('out.dtype must be int')
        if out.shape != out_shape:
            raise ValueError('shape mismatch')
    ca.argmax(one_hot, axis=1, out=out)
    return out

# ---------------------- Extensions ---------------------

# map_from doesn't work because of concurrancy issues -- disable it!
#def copy_rows(rowids, from_mat, to_mat):
def copy_rows(rowids, from_mat, to_mat, map_from=True):
    """
    If map_from is True this can be implemented in numpy via:

    to_mat=from_mat[rowids]

    else its

    to_mat[rowids]=from_mat

    """
    if map_from:
        _shape = (rowids.size, from_mat.shape[1])
        if to_mat.shape != _shape:
            raise ValueError('shape mismatch: %s != %s'%(to_mat.shape, _shape))
    else:
        _shape = (rowids.size, to_mat.shape[1])
        if from_mat.shape != _shape:
            raise ValueError('shape mismatch: %s != %s'%(from_mat.shape,
                _shape))
    if rowids.dtype != np.dtype('int32'):
        raise ValueError('rowids.dtype must be int')
    mapfrom=1 if map_from else 0
    nnet._copy_rows(rowids._data, _shape[0], _shape[1], from_mat._data, 
            to_mat._data, mapfrom)
    return to_mat



def copy_sum_rows(rowids, from_mat, to_mat, map_from=True, coefficients=None,
        constant=1., var=1.):
    """
    If map_from is True we're basically implementing:

    to_mat[i,k] = sum_j coefficients[i,j] * from_mat[rowids[i,j],k]

    else we're implementing:

    for j:
        to_mat[rowids[i,j], k] += coefficient[i,j] * from_mat[i,k]

    Note if coefficients is None we replace it by the matrix:

     coefficient[i,j] =  constant * var**j 

    the default values for these are 1.

    """
    constant=float(constant)
    var=float(var)
    to_shape = (rowids.shape[0], from_mat.shape[1])
    if rowids.dtype != np.dtype('int32'):
        raise ValueError('rowids.dtype must be int')
    if map_from:
        if to_mat.shape != to_shape:
            raise ValueError('shape mismatch rowids, to_mat: %s vs %s'%(
                to_mat.shape, to_shape))
    if coefficients is not None:
        if rowids.shape != coefficients.shape:
            raise ValueError('shape mismatch rowids, coefficients')
        if coefficients.dtype != np.dtype('float32'):
            raise ValueError('coefficients must be floats')
        coefdata=coefficients._data
    else:
        coefdata=None
    numsum=rowids.shape[1]
    mapfrom=1 if map_from else 0
    nnet._copy_sum_rows(rowids._data, numsum, to_shape[0], to_shape[1],
            from_mat._data, to_mat._data, mapfrom, coefdata, constant, var)
    return to_mat

