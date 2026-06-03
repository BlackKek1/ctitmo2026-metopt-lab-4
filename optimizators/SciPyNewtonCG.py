import numpy as np

from scipy.optimize import minimize
from functions.BaseFunction import BaseFunction
from optimizators.BaseOptimizer import BaseOptimizer
from optimizators.OptimizationResult import OptimizationResult


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
