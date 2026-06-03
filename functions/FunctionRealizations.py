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
