from optimizators.BaseOptimizer import BaseOptimizer
from optimizators.OptimazerUtils import armijo


class FletcherReevesCG(BaseOptimizer):
  def __init__(self):
    super().__init__('Fletcher-Reeves')
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
    a = armijo(func, x, g, d)
    self._g_prev = g.copy()
    self._d_prev = d.copy()
    return a * d, ''
