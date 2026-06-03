import numpy as np

from functions.FunctionRealizations import (
  make_quadratic,
  Rosenbrock,
  Himmelblau,
  Ackley,
)
from optimizators.OptimizatorsRealizations import (
  QuadraticConjugateGradient,
  FletcherReevesCG,
  PolakRibiereCG,
  NewtonCholesky,
  NewtonDirectionChoice,
  PowellDogLeg,
  DFP,
  BFGS,
  LBFGS,
  SciPyNewtonCG,
)
from processing.Calculation import Calculation


def main():
  np.set_printoptions(precision=4, suppress=True)
  np.seterr(over='ignore')

  _EPS = 1e-8

  quad_optimizers = [
    QuadraticConjugateGradient(),
    FletcherReevesCG(),
    PolakRibiereCG(),
    NewtonCholesky(),
    NewtonDirectionChoice(),
    PowellDogLeg(),
    DFP(),
    BFGS(),
    LBFGS(m=5),
    SciPyNewtonCG(),
  ]

  nonlinear_optimizers = [
    FletcherReevesCG(),
    PolakRibiereCG(),
    NewtonCholesky(),
    NewtonDirectionChoice(),
    PowellDogLeg(),
    DFP(),
    BFGS(),
    LBFGS(m=5),
    SciPyNewtonCG(),
  ]

  quadratic_2d = make_quadratic(2, 10, seed=0, name='quadratic 2d (k=10)')

  rosenbrock = Rosenbrock()
  himmelblau = Himmelblau()
  ackley = Ackley()

  functions = [rosenbrock, himmelblau, ackley]

  starts_2d = [
    np.array([4.0, 4.0]),
    np.array([-4.0, 3.0]),
    np.array([3.0, -4.0]),
    np.array([-5.0, -2.0]),
    np.array([1.5, -3.5]),
  ]

  starts = {
    rosenbrock.name: [
      np.array([-1.2, 1.0]),
      np.array([2.0, 2.0]),
      np.array([-2.0, 2.0]),
      np.array([1.5, -1.0]),
      np.array([0.0, 0.0]),
    ],
    himmelblau.name: [
      np.array([0.0, 0.0]),
      np.array([-4.0, 0.0]),
      np.array([4.0, 4.0]),
      np.array([-3.0, 3.0]),
      np.array([5.0, -5.0]),
    ],
    ackley.name: [
      np.array([2.5, 2.5]),
      np.array([2.0, -2.0]),
      np.array([-3.0, 1.0]),
      np.array([4.0, -4.0]),
      np.array([-5.0, -5.0]),
    ],
  }

  contour_ranges = {
    quadratic_2d.name: ((-6.0, 6.0), (-6.0, 6.0)),
    rosenbrock.name: ((-2.5, 2.5), (-1.5, 3.0)),
    himmelblau.name: ((-6.0, 6.0), (-6.0, 6.0)),
    ackley.name: ((-6.0, 6.0), (-6.0, 6.0)),
  }

  Calculation.quadratic_grid_experiment(
    optimizers=quad_optimizers,
    ns=[2, 10, 50, 100],
    ks=[1, 10, 100, 1000],
    eps=_EPS,
    seeds=[0, 1, 2],
  )

  Calculation.quadratic_start_points_experiment(
    func=quadratic_2d,
    optimizers=quad_optimizers,
    starts=starts_2d,
    eps=_EPS,
    contour_range=contour_ranges[quadratic_2d.name],
  )

  Calculation.complex_functions_experiment(
    functions=functions,
    optimizers=nonlinear_optimizers,
    starts=starts,
    eps=_EPS,
    contour_ranges=contour_ranges,
  )

  Calculation.lbfgs_memory_experiment(
    func=make_quadratic(
      n=10,
      k=15,
      seed=0
    ),
    x0=np.full(10, 4.0),
    ms=[2, 5, 10, 20, 40],
    eps=_EPS,
  )

if __name__ == '__main__':
  main()
