import numpy as np

from optimizators.BaseOptimizer import BaseOptimizer
from optimizators.OptimazerUtils import armijo


class BFGS(BaseOptimizer):
  def __init__(self):
    super().__init__('BFGS')
    self.h = None

  def _reset(self):
    self.h = None

  def _step(self, func, x, g):
    if self.h is None:
      self.h = np.eye(len(x))
    d = -self.h @ g
    a = armijo(func, x, g, d)
    self._x_old = x.copy()
    self._g_old = g.copy()
    return a * d, ''

  def _after_step(self, func, x, g):
    s = x - self._x_old
    y = g - self._g_old
    sy = float(s @ y)
    if sy > 1e-12:
      rho = 1.0 / sy
      i_mat = np.eye(len(x))
      v = i_mat - rho * np.outer(s, y)
      self.h = v @ self.h @ v.T + rho * np.outer(s, s)
