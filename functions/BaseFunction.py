import numpy as np
from abc import ABC, abstractmethod


class BaseFunction(ABC):
  def __init__(self, name: str):
    self.name = name
    self.func_calls = 0
    self.grad_calls = 0
    self.hess_calls = 0

  def __call__(self, x: np.ndarray) -> float:
    self.func_calls += 1
    return self._compute_value(np.asarray(x, dtype=float).ravel())

  def grad(self, x: np.ndarray) -> np.ndarray:
    self.grad_calls += 1
    return self._compute_grad(np.asarray(x, dtype=float).ravel())

  def hess(self, x: np.ndarray) -> np.ndarray:
    self.hess_calls += 1
    return self._compute_hess(np.asarray(x, dtype=float).ravel())

  def reset_counters(self):
    self.func_calls = 0
    self.grad_calls = 0
    self.hess_calls = 0

  def compute_without_increment(self, x: np.ndarray) -> float:
    return self._compute_value(np.asarray(x, dtype=float).ravel())

  def compute_grad_without_increment(self, x: np.ndarray) -> np.ndarray:
    return self._compute_grad(np.asarray(x, dtype=float).ravel())

  @abstractmethod
  def _compute_value(self, x: np.ndarray) -> float:
    raise NotImplementedError

  @abstractmethod
  def _compute_grad(self, x: np.ndarray) -> np.ndarray:
    raise NotImplementedError

  @abstractmethod
  def _compute_hess(self, x: np.ndarray) -> np.ndarray:
    raise NotImplementedError
