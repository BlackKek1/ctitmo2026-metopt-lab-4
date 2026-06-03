import numpy as np

from optimizators.BaseOptimizer import BaseOptimizer


class QuadraticConjugateGradient(BaseOptimizer):
  def __init__(self):
    super().__init__('Conjugate gradients')
    self._g_prev = None
    self._d_prev = None

  def _reset(self):
    self._g_prev = None
    self._d_prev = None

  def _step(self, func, x, g):
    if self._g_prev is None:
      d = -g
    else:
      beta = float(g @ g) / float(self._g_prev @ self._g_prev)
      d = -g + beta * self._d_prev
      if g @ d >= 0:
        d = -g
    h = func.hess(x)
    denom = float(d @ h @ d)
    if denom <= 0:
      return np.zeros_like(x), 'nonpositive curvature'
    a = -float(g @ d) / denom
    self._g_prev = g.copy()
    self._d_prev = d.copy()
    return a * d, ''