from abc import ABC, abstractmethod
import numpy as np

from functions.BaseFunction import BaseFunction
from optimizators.OptimizationResult import OptimizationResult


class BaseOptimizer(ABC):
  def __init__(self, name: str, max_iter: int = 1000):
    self.name = name
    self.max_iter = max_iter

  def _reset(self):
    return None

  def _after_step(self, func: BaseFunction, x: np.ndarray, g: np.ndarray):
    return None

  def optimize(self, func: BaseFunction, x0: np.ndarray, epsilon: float = 1e-8) -> OptimizationResult:
    self._reset()
    func.reset_counters()

    x = np.array(x0, dtype=float, copy=True)
    traj = [x.copy()]
    g = func.grad(x)
    it = 0
    status = 'Max iteration reached'

    while np.linalg.norm(g) > epsilon and it < self.max_iter:
      p, new_status = self._step(func, x, g)
      if new_status != '':
        status = new_status
        break
      x = x + p
      traj.append(x.copy())
      g = func.grad(x)
      it += 1
      self._after_step(func, x, g)

    if np.linalg.norm(g) <= epsilon:
      status = 'Gradient tolerance'

    return OptimizationResult(x, func.compute_without_increment(x), it, func.func_calls, func.grad_calls,
                              func.hess_calls, status, np.array(traj))

  @abstractmethod
  def _step(self, func: BaseFunction, x: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, str]:
    raise NotImplementedError
