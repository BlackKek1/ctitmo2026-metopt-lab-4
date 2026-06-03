import numpy as np

from functions.BaseFunction import BaseFunction


class Himmelblau(BaseFunction):
  def __init__(self):
    super().__init__('Himmelblau')

  def _compute_value(self, x: np.ndarray) -> float:
    return ((x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2).item()

  def _compute_grad(self, x: np.ndarray) -> np.ndarray:
    dx0 = 4 * x[0] * (x[0] ** 2 + x[1] - 11) + 2 * (x[0] + x[1] ** 2 - 7)
    dx1 = 2 * (x[0] ** 2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1] ** 2 - 7)
    return np.array([dx0, dx1])

  def _compute_hess(self, x: np.ndarray) -> np.ndarray:
    return np.array([
      [12 * x[0] ** 2 + 4 * x[1] - 42, 4 * x[0] + 4 * x[1]],
      [4 * x[0] + 4 * x[1], 12 * x[1] ** 2 + 4 * x[0] - 26]
    ])
