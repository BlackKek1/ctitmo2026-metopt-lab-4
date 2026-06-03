from dataclasses import dataclass
import numpy as np


@dataclass
class OptimizationResult:
  x_opt: np.ndarray
  f_opt: float
  iterations: int
  func_calls: int
  grad_calls: int
  hess_calls: int
  status: str
  trajectory: np.ndarray
