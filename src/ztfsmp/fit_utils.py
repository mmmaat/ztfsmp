#!/usr/bin/env python3

import numpy as np
from scipy import sparse
from sksparse import cholmod
import pandas as pd
from numpy.polynomial.polynomial import Polynomial

import imageproc.composable_functions as compfuncs
import saunerie.fitparameters as fp
from saunerie.linearmodels import indic, RobustLinearSolver
from croaks import DataProxy


class BiPol2DModel():
    def __init__(self, degree, space_count=1):
        self.bipol2d = compfuncs.BiPol2D(deg=degree, key='space', n=space_count)

        self.params = fp.FitParameters([*self.bipol2d.get_struct()])

    def __get_coeffs(self):
        return dict((key, self.params[key].full) for key in self.params._struct.slices.keys())

    def __set_coeffs(self, coeffs):
        for key in coeffs.keys():
            self.params[key] = coeffs[key]

    coeffs = property(__get_coeffs, __set_coeffs)

    def __call__(self, x, p=None, space_indices=None, jac=False):
        if p is not None:
            self.params.free = p

        if space_indices is None:
            space_indices = [0]*x.shape[1]

        # Evaluate polynomials
        if not jac:
            return self.bipol2d(x, self.params, space=space_indices)
        else:
            xy, _, (i, j, vals) = self.bipol2d.derivatives(x, self.params, space=space_indices)

            ii = np.hstack([i, i+x[0].shape[0]])
            jj = np.tile(j, 2).ravel()
            vv = np.hstack(vals).ravel()

            ii = np.hstack(ii)
            jj = np.hstack(jj)
            vv = np.hstack(vv)

            J_model = sparse.coo_array((vv, (ii, jj)), shape=(2*xy.shape[1], len(self.params.free)))

            return xy, J_model

    def fit(self, x, y, space_indices=None):
        _, J = self.__call__(x, space_indices=space_indices, jac=True)
        H = J.T @ J
        B = J.T @ np.hstack(y)

        fact = cholmod.cholesky(H.tocsc())
        p = fact(B)
        self.params.free = p
        return p

    def residuals(self, x, y, space_indices=None):
        y_model = self.__call__(x, space_indices=space_indices)
        return y_model - y


def BiPol2D_fit(x, y, degree, space_indices=None, simultaneous_fit=True):
    if space_indices is None:
        space_count = 1
    else:
        space_count = len(set(space_indices))

    if simultaneous_fit:
        model = BiPol2DModel(degree, space_count=space_count)
        model.fit(x, y, space_indices=space_indices)
    else:
        space_models_coeffs = []
        for space_index in range(space_count):
            space_model = BiPol2DModel(degree, space_count=1)
            space_model.fit(x[:, space_indices==space_index], y[:, space_indices==space_index])
            space_models_coeffs.append(space_model.coeffs)

        model = BiPol2DModel(degree, space_count=space_count)
        for space_index in range(space_count):
            for coeff in model.coeffs.keys():
                model.coeffs[coeff][space_index] = space_models_coeffs[space_index][coeff]

    return model


def RobustPolynomialFit(x, y, degree, dy=None, just_chi2=False):
    assert len(x) == len(y)
    model = None

    dp = DataProxy(pd.DataFrame({'x': x, 'y': x}).to_records(), x='x', y='y')

    for i in range(degree+1):
        deg_model = indic([0]*len(x), val=x**i, name='deg{}'.format(i))
        if model is None:
            model = deg_model
        else:
            model = model + deg_model

    if dy is not None:
        solver = RobustLinearSolver(model, y, weights=1./dy)
    else:
        solver = RobustLinearSolver(model, dp.y)

    solver.model.params.free = solver.robust_solution()
    res = solver.get_res(y)
    if dy is not None:
        wres = res/dy
    else:
        wres = res

    chi2 = sum(wres**2)/(len(x)-degree-1)
    if just_chi2:
        return chi2
    else:
        return Polynomial(solver.model.params.free[:]), chi2
        # return solver.model.params.free[:], chi2
