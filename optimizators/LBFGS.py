from optimizators.BaseOptimizer import BaseOptimizer
from optimizators.OptimazerUtils import armijo


class LBFGS(BaseOptimizer):
  def __init__(self, m: int = 5):
    super().__init__(f'L-BFGS(m={m})')
    self.m = m
    self.s = []
    self.y = []

  def _reset(self):
    self.s = []
    self.y = []

  def _direction(self, g):
    q = g.copy()
    alpha = []
    for s, y in zip(reversed(self.s), reversed(self.y)):
      r = 1.0 / float(y @ s)
      a = r * float(s @ q)
      alpha.append(a)
      q = q - a * y
    if self.s:
      s, y = self.s[-1], self.y[-1]
      gamma = float(s @ y) / float(y @ y)
      r = gamma * q
    else:
      r = q
    for (s, y), a in zip(zip(self.s, self.y), reversed(alpha)):
      r += s * (a - float(y @ r) / float(y @ s))
    return -r

  def _step(self, func, x, g):
    d = self._direction(g)
    if g @ d >= 0:
      d = -g
    a = armijo(func, x, g, d)
    self._x_old = x.copy()
    self._g_old = g.copy()
    return a * d, ''

  def _after_step(self, func, x, g):
    s = x - self._x_old
    y = g - self._g_old
    if float(s @ y) > 1e-12:
      self.s.append(s.copy())
      self.y.append(y.copy())
      self.s = self.s[-self.m:]
      self.y = self.y[-self.m:]
