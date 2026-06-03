import numpy as np

from functions.BaseFunction import BaseFunction


def armijo(func: BaseFunction, x: np.ndarray, g: np.ndarray, d: np.ndarray, a0: float = 1.0, c: float = 1e-4,
           rho: float = 0.5) -> float:
  fx = func.compute_without_increment(x)
  gd = float(g @ d)
  a = a0
  while func(x + a * d) > fx + c * a * gd:
    a *= rho
    if a < 1e-16:
      break
  return a
