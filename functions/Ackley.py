import numpy as np

from functions.BaseFunction import BaseFunction


class Ackley(BaseFunction):
  def __init__(self):
    super().__init__('Ackley')

  def _compute_value(self, x: np.ndarray) -> float:
    t1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2)))
    t2 = -np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1])))
    return (t1 + t2 + np.e + 20).item()

  def _compute_grad(self, x: np.ndarray) -> np.ndarray:
    r_sq = np.maximum(0.5 * (x[0] ** 2 + x[1] ** 2), 1e-12)
    r = np.sqrt(r_sq)

    t1 = np.exp(-0.2 * r)
    t2 = np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1])))

    dx0 = 20 * 0.2 * t1 * (0.5 * x[0] / r) + t2 * 0.5 * 2 * np.pi * np.sin(2 * np.pi * x[0])
    dx1 = 20 * 0.2 * t1 * (0.5 * x[1] / r) + t2 * 0.5 * 2 * np.pi * np.sin(2 * np.pi * x[1])
    return np.array([dx0, dx1])

  def _compute_hess(self, x: np.ndarray) -> np.ndarray:
    r_sq = 0.5 * (x[0] ** 2 + x[1] ** 2)

    if r_sq < 1e-12: return np.eye(2) * (2.0 * np.pi ** 2)

    r = np.sqrt(r_sq)
    t1 = np.exp(-0.2 * r)
    t2 = np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1])))
    h1_00 = t1 * (2.0 / r - 0.2 * (x[0] ** 2) / (r ** 2) - (x[0] ** 2) / (r ** 3))
    h1_11 = t1 * (2.0 / r - 0.2 * (x[1] ** 2) / (r ** 2) - (x[1] ** 2) / (r ** 3))
    h1_01 = t1 * x[0] * x[1] * (-0.2 / (r ** 2) - 1.0 / (r ** 3))
    h2_00 = (np.pi ** 2) * t2 * (np.sin(2 * np.pi * x[0]) ** 2 + 2 * np.cos(2 * np.pi * x[0]))
    h2_11 = (np.pi ** 2) * t2 * (np.sin(2 * np.pi * x[1]) ** 2 + 2 * np.cos(2 * np.pi * x[1]))
    h2_01 = (np.pi ** 2) * t2 * np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1])

    hessian = np.zeros((2, 2))
    hessian[0, 0] = h1_00 + h2_00
    hessian[1, 1] = h1_11 + h2_11
    hessian[0, 1] = h1_01 + h2_01
    hessian[1, 0] = hessian[0, 1]

    return hessian
