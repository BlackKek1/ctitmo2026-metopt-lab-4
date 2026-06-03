import numpy as np

from optimizators.BaseOptimizer import BaseOptimizer
from functions.BaseFunction import BaseFunction
from numpy.linalg import LinAlgError
from optimizators.OptimizationResult import OptimizationResult


class PowellDogLeg(BaseOptimizer):
  def __init__(self, delta0: float = 1.0, delta_max: float = 100.0, eta: float = 0.1):
    super().__init__('Powell Dog Leg')
    self.delta0 = delta0
    self.delta_max = delta_max
    self.eta = eta
    self.delta = delta0

  def _reset(self):
    self.delta = self.delta0

  def _step(self, func, x, g):
    return np.zeros_like(x), ''

  def optimize(self, func: BaseFunction, x0: np.ndarray, epsilon: float = 1e-8):
    self._reset()
    func.reset_counters()
    x = np.array(x0, dtype=float, copy=True)
    traj = [x.copy()]
    g = func.grad(x)
    it = 0
    status = 'max_iter'

    while np.linalg.norm(g) > epsilon and it < self.max_iter:
      h = func.hess(x)

      try:
        l = np.linalg.cholesky(h)
        y = np.linalg.solve(l, g)
        p_b = -np.linalg.solve(l.T, y)
      except LinAlgError:
        p_b = -g

      ghg = float(g @ h @ g)
      if ghg > 0:
        p_u = -(float(g @ g) / ghg) * g
      else:
        p_u = -(self.delta / np.linalg.norm(g)) * g

      if np.linalg.norm(p_b) <= self.delta:
        p = p_b
      elif np.linalg.norm(p_u) >= self.delta:
        p = self.delta / np.linalg.norm(p_u) * p_u
      else:
        d = p_b - p_u
        a = float(d @ d)
        b = 2 * float(p_u @ d)
        c = float(p_u @ p_u) - self.delta ** 2
        tau = (-b + np.sqrt(max(b * b - 4 * a * c, 0.0))) / (2 * a)
        p = p_u + tau * d

      fx = func.compute_without_increment(x)
      mp = fx + g @ p + 0.5 * p @ h @ p
      fnew = func(x + p)
      rho = (fx - fnew) / (fx - mp) if abs(fx - mp) > 1e-16 else 0.0

      if rho < 0.25:
        self.delta *= 0.25
      elif rho > 0.75 and np.linalg.norm(p) >= 0.99 * self.delta:
        self.delta = min(2 * self.delta, self.delta_max)

      if rho > self.eta:
        x = x + p
        traj.append(x.copy())
        g = func.grad(x)
      it += 1

    if np.linalg.norm(g) <= epsilon:
      status = 'gradient tolerance'

    return OptimizationResult(x, func.compute_without_increment(x), it, func.func_calls, func.grad_calls,
                              func.hess_calls, status, np.array(traj))
