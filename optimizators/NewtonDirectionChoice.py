import numpy as np

from numpy.linalg import LinAlgError
from optimizators.BaseOptimizer import BaseOptimizer
from optimizators.OptimazerUtils import armijo


class NewtonDirectionChoice(BaseOptimizer):
  def __init__(self):
    super().__init__('Newton direction choice')

  def _step(self, func, x, g):
    h = func.hess(x)
    try:
      np.linalg.cholesky(h)
      d = -np.linalg.solve(h, g)
    except LinAlgError:
      d = -g
    return armijo(func, x, g, d) * d, ''
