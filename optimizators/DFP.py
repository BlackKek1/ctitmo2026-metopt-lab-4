import numpy as np

from optimizators.BaseOptimizer import BaseOptimizer
from optimizators.OptimazerUtils import armijo


class DFP(BaseOptimizer):
  def __init__(self):
    super().__init__('DFP')
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
      hy = self.h @ y
      yhy = float(y @ hy)
      if yhy > 1e-12:
        self.h = self.h + np.outer(s, s) / sy - np.outer(hy, hy) / yhy
