import numpy as np

from functions.BaseFunction import BaseFunction


class Rosenbrock(BaseFunction):
  def __init__(self):
    super().__init__('Rosenbrock')

  def _compute_value(self, x: np.ndarray) -> float:
    return ((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2).item()

  def _compute_grad(self, x: np.ndarray) -> np.ndarray:
    dx0 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
    dx1 = 200 * (x[1] - x[0] ** 2)
    return np.array([dx0, dx1])

  def _compute_hess(self, x: np.ndarray) -> np.ndarray:
    return np.array([
      [2 - 400 * x[1] + 1200 * x[0] ** 2, -400 * x[0]],
      [-400 * x[0], 200.0]
    ])
