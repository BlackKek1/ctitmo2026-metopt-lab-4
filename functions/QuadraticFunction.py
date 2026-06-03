import numpy as np

from functions.BaseFunction import BaseFunction


class QuadraticFunction(BaseFunction):
  def __init__(self, a: np.ndarray, b: np.ndarray, c: float = 0.0, x_star: np.ndarray | None = None,
               name: str = 'Quadratic'):
    super().__init__(name)
    self.a = np.asarray(a, dtype=float)
    self.b = np.asarray(b, dtype=float)
    self.c = float(c)
    self.x_star = None if x_star is None else np.asarray(x_star, dtype=float).ravel()

  def _compute_value(self, x: np.ndarray) -> float:
    return float(0.5 * x @ self.a @ x - self.b @ x + self.c)

  def _compute_grad(self, x: np.ndarray) -> np.ndarray:
    return self.a @ x - self.b

  def _compute_hess(self, x: np.ndarray) -> np.ndarray:
    return self.a


def make_quadratic(n: int, k: float = 10.0, seed: int = 0, name: str | None = None) -> QuadraticFunction:
  rng = np.random.default_rng(seed)
  q, _ = np.linalg.qr(rng.normal(size=(n, n)))
  lam = np.logspace(0, np.log10(k), n) if n > 1 else np.array([k], dtype=float)
  a = q.T @ np.diag(lam) @ q
  x_star = rng.normal(size=n)
  b = a @ x_star
  c = 0.5 * x_star @ a @ x_star
  return QuadraticFunction(a, b, c, x_star=x_star, name=name or f'Quadratic n={n}, k={k}')


class QuadFuncExample(QuadraticFunction):
  def __init__(self):
    a = np.array([[2.0, 0.0], [0.0, 20.0]])
    x_star = np.array([0.0, 0.0])
    super().__init__(a, a @ x_star, 0.0, x_star=x_star, name='Quad Example (k=10)')
