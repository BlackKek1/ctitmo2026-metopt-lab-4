import numpy as np

from optimizators.BaseOptimizer import BaseOptimizer
from numpy.linalg import LinAlgError


class NewtonCholesky(BaseOptimizer):
  def __init__(self):
    super().__init__('Newton-Cholesky')

  def _step(self, func, x, g):
    h = func.hess(x)
    try:
      l = np.linalg.cholesky(h)
      y = np.linalg.solve(l, -g)
      p = np.linalg.solve(l.T, y)
      return p, ''
    except LinAlgError:
      return np.zeros_like(x), 'Hessian is not positive definite'
