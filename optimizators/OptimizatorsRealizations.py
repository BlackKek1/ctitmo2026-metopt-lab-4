import numpy as np
from numpy.linalg import LinAlgError
from scipy.optimize import minimize

from functions.BaseFunction import BaseFunction
from optimizators.BaseOptimizer import BaseOptimizer
from optimizators.OptimizationResult import OptimizationResult


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


class QuadraticConjugateGradient(BaseOptimizer):
  def __init__(self):
    super().__init__('Conjugate gradients')
    self._g_prev = None
    self._d_prev = None

  def _reset(self):
    self._g_prev = None
    self._d_prev = None

  def _step(self, func, x, g):
    if self._g_prev is None:
      d = -g
    else:
      beta = float(g @ g) / float(self._g_prev @ self._g_prev)
      d = -g + beta * self._d_prev
      if g @ d >= 0:
        d = -g
    h = func.hess(x)
    denom = float(d @ h @ d)
    if denom <= 0:
      return np.zeros_like(x), 'nonpositive curvature'
    a = -float(g @ d) / denom
    self._g_prev = g.copy()
    self._d_prev = d.copy()
    return a * d, ''


class FletcherReevesCG(BaseOptimizer):
  def __init__(self):
    super().__init__('Fletcher-Reeves')
    self._g_prev = None
    self._d_prev = None

  def _reset(self):
    self._g_prev = None
    self._d_prev = None

  def _step(self, func, x, g):
    if self._g_prev is None:
      d = -g
    else:
      beta = float(g @ g) / float(self._g_prev @ self._g_prev)
      d = -g + beta * self._d_prev
      if g @ d >= 0:
        d = -g
    a = armijo(func, x, g, d)
    self._g_prev = g.copy()
    self._d_prev = d.copy()
    return a * d, ''


class PolakRibiereCG(BaseOptimizer):
  def __init__(self):
    super().__init__('Polak-Ribiere')
    self._g_prev = None
    self._d_prev = None

  def _reset(self):
    self._g_prev = None
    self._d_prev = None

  def _step(self, func, x, g):
    if self._g_prev is None:
      d = -g
    else:
      beta = max(0.0, float(g @ (g - self._g_prev)) / float(self._g_prev @ self._g_prev))
      d = -g + beta * self._d_prev
      if g @ d >= 0:
        d = -g
    a = armijo(func, x, g, d)
    self._g_prev = g.copy()
    self._d_prev = d.copy()
    return a * d, ''


class NewtonCholesky(BaseOptimizer):
  def __init__(self):
    super().__init__('Newton-Cholesky')

  def _step(self, func, x, g):
    h = func.hess(x)
    try:
      l = np.linalg.cholesky(h)
      y = np.linalg.solve(l, -g)
      p = np.linalg.solve(l.T, y)
      return p, ''
    except LinAlgError:
      return np.zeros_like(x), 'Hessian is not positive definite'


class NewtonDirectionChoice(BaseOptimizer):
  def __init__(self):
    super().__init__('Newton direction choice')

  def _step(self, func, x, g):
    h = func.hess(x)
    try:
      np.linalg.cholesky(h)
      d = -np.linalg.solve(h, g)
    except LinAlgError:
      d = -g
    return armijo(func, x, g, d) * d, ''


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


class DFP(BaseOptimizer):
  def __init__(self):
    super().__init__('DFP')
    self.h = None

  def _reset(self):
    self.h = None

  def _step(self, func, x, g):
    if self.h is None:
      self.h = np.eye(len(x))
    d = -self.h @ g
    a = armijo(func, x, g, d)
    self._x_old = x.copy()
    self._g_old = g.copy()
    return a * d, ''

  def _after_step(self, func, x, g):
    s = x - self._x_old
    y = g - self._g_old
    sy = float(s @ y)
    if sy > 1e-12:
      hy = self.h @ y
      yhy = float(y @ hy)
      if yhy > 1e-12:
        self.h = self.h + np.outer(s, s) / sy - np.outer(hy, hy) / yhy


class BFGS(BaseOptimizer):
  def __init__(self):
    super().__init__('BFGS')
    self.h = None

  def _reset(self):
    self.h = None

  def _step(self, func, x, g):
    if self.h is None:
      self.h = np.eye(len(x))
    d = -self.h @ g
    a = armijo(func, x, g, d)
    self._x_old = x.copy()
    self._g_old = g.copy()
    return a * d, ''

  def _after_step(self, func, x, g):
    s = x - self._x_old
    y = g - self._g_old
    sy = float(s @ y)
    if sy > 1e-12:
      rho = 1.0 / sy
      i_mat = np.eye(len(x))
      v = i_mat - rho * np.outer(s, y)
      self.h = v @ self.h @ v.T + rho * np.outer(s, s)


class LBFGS(BaseOptimizer):
  def __init__(self, m: int = 5):
    super().__init__(f'L-BFGS(m={m})')
    self.m = m
    self.s = []
    self.y = []

  def _reset(self):
    self.s = []
    self.y = []

  def _direction(self, g):
    q = g.copy()
    alpha = []
    for s, y in zip(reversed(self.s), reversed(self.y)):
      r = 1.0 / float(y @ s)
      a = r * float(s @ q)
      alpha.append(a)
      q = q - a * y
    if self.s:
      s, y = self.s[-1], self.y[-1]
      gamma = float(s @ y) / float(y @ y)
      r = gamma * q
    else:
      r = q
    for (s, y), a in zip(zip(self.s, self.y), reversed(alpha)):
      r += s * (a - float(y @ r) / float(y @ s))
    return -r

  def _step(self, func, x, g):
    d = self._direction(g)
    if g @ d >= 0:
      d = -g
    a = armijo(func, x, g, d)
    self._x_old = x.copy()
    self._g_old = g.copy()
    return a * d, ''

  def _after_step(self, func, x, g):
    s = x - self._x_old
    y = g - self._g_old
    if float(s @ y) > 1e-12:
      self.s.append(s.copy())
      self.y.append(y.copy())
      self.s = self.s[-self.m:]
      self.y = self.y[-self.m:]


class SciPyNewtonCG(BaseOptimizer):
  def __init__(self):
    super().__init__('Newton-CG')

  def _step(self, func, x, g):
    raise NotImplementedError("Прямой вызов scipy.optimize")

  def optimize(self, func: BaseFunction, x0: np.ndarray, epsilon: float = 1e-8) -> OptimizationResult:
    func.reset_counters()
    traj = [np.array(x0, copy=True)]

    res = minimize(
      fun=func,
      x0=x0,
      method='Newton-CG',
      jac=func.grad,
      hess=func.hess,
      callback=lambda xk: traj.append(xk.copy()),
      options={'xtol': epsilon, 'maxiter': self.max_iter}
    )

    return OptimizationResult(
      x_opt=res.x,
      f_opt=func.compute_without_increment(res.x),
      iterations=res.nit,
      func_calls=func.func_calls,
      grad_calls=func.grad_calls,
      hess_calls=func.hess_calls,
      status=res.message,
      trajectory=np.array(traj)
    )
