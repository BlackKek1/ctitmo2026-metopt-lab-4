import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

from functions.BaseFunction import BaseFunction
from functions.QuadraticFunction import make_quadratic
from optimizators.BaseOptimizer import BaseOptimizer
from optimizators.LBFGS import LBFGS


class Calculation:
  def __new__(cls, *args, **kwargs):
    raise TypeError('Cannot create object')

  @staticmethod
  def print_table(rows, title: str | None = None):
    if title:
      print('\n' + title)
    print(
      tabulate(
        rows,
        headers='keys',
        tablefmt='grid',
        showindex=False,
        floatfmt='.10f',
        stralign='left',
        numalign='right',
      )
    )

  @staticmethod
  def _to_float(x):
    return float(np.asarray(x, dtype=float))

  @staticmethod
  def _format_vec(x):
    x = np.asarray(x, dtype=float).ravel()
    return '[' + ', '.join(f'{v:.6f}' for v in x) + ']'

  @staticmethod
  def _run(opt: BaseOptimizer, func: BaseFunction, x0: np.ndarray, eps: float):
    return opt.optimize(func, np.asarray(x0, dtype=float), epsilon=eps)

  @staticmethod
  def _row(func: BaseFunction, opt: BaseOptimizer, res, **extra):
    x_opt = np.asarray(getattr(res, 'x_opt', np.array([])), dtype=float)
    status = getattr(res, 'status', '') or 'unknown'

    row = {
      'function': func.name,
      'optimizer': opt.name,
      'it': int(getattr(res, 'iterations', 0)),
      'f_calls': int(getattr(res, 'func_calls', 0)),
      'g_calls': int(getattr(res, 'grad_calls', 0)),
      'h_calls': int(getattr(res, 'hess_calls', 0)),
      'status': str(status).lower(),
      'f_opt': Calculation._to_float(getattr(res, 'f_opt', np.nan)),
      'grad_norm': Calculation._to_float(
        np.linalg.norm(func.compute_grad_without_increment(x_opt))) if x_opt.size else np.nan,
      'x_opt': Calculation._format_vec(x_opt) if x_opt.size else '',
    }

    row.update(extra)
    return row

  @staticmethod
  def _aggregate(rows, keys):
    grouped = {}
    for r in rows:
      k = tuple(r[name] for name in keys)
      grouped.setdefault(k, []).append(r)

    result = []
    metrics = ['it', 'f_calls', 'g_calls', 'h_calls', 'f_opt', 'grad_norm']

    for k, items in grouped.items():
      row = {name: k[i] for i, name in enumerate(keys)}
      for metric in metrics:
        vals = [r[metric] for r in items if metric in r and r[metric] is not None and not np.isnan(r[metric])]
        if vals:
          row[metric] = float(np.mean(vals))
      row['status'] = max(
        set(r['status'] for r in items),
        key=lambda s: sum(rr['status'] == s for rr in items),
      )
      result.append(row)

    return result

  @staticmethod
  def _optimizer_summary(rows):
    summary = []
    for name in sorted(set(r['optimizer'] for r in rows)):
      cur = [r for r in rows if r['optimizer'] == name]
      summary.append({
        'optimizer': name,
        'avg_it': float(np.mean([r['it'] for r in cur])),
        'avg_f_calls': float(np.mean([r['f_calls'] for r in cur])),
        'avg_g_calls': float(np.mean([r['g_calls'] for r in cur])),
        'avg_h_calls': float(np.mean([r['h_calls'] for r in cur])),
        'avg_cost': float(np.mean([r['f_calls'] + r['g_calls'] + r['h_calls'] for r in cur])),
      })
    summary.sort(key=lambda r: r['avg_cost'])
    return summary

  @staticmethod
  def _plot_metric_series(rows, x_key: str, x_label: str, title_prefix: str):
    metrics = [
      ('it', 'iterations'),
      ('f_calls', 'function calls'),
      ('g_calls', 'gradient calls'),
      ('h_calls', 'hessian calls'),
    ]

    optimizers = sorted(set(r['optimizer'] for r in rows))

    for metric_key, metric_name in metrics:
      plt.figure(figsize=plt.rcParams['figure.figsize'])

      for opt_name in optimizers:
        cur = [r for r in rows if r['optimizer'] == opt_name]
        if not cur:
          continue
        cur.sort(key=lambda r: r[x_key])
        xs = [r[x_key] for r in cur]
        ys = [r[metric_key] for r in cur]
        plt.plot(xs, ys, marker='o', label=opt_name)

      plt.title(f'{title_prefix}: {metric_name}')
      plt.xlabel(x_label)
      plt.ylabel(metric_name)
      plt.grid(True, linestyle=':', alpha=0.5)
      plt.legend(fontsize=8)
      plt.tight_layout()
      plt.show()

  @staticmethod
  def _plot_contour_trajectory(func: BaseFunction, trajectory: np.ndarray, title: str, x_range, y_range):
    traj = np.asarray(trajectory, dtype=float)
    if traj.size == 0:
      return

    xs = np.linspace(x_range[0], x_range[1], 240)
    ys = np.linspace(y_range[0], y_range[1], 240)
    x, y = np.meshgrid(xs, ys)

    z = np.array([
      [func.compute_without_increment(np.array([x, y], dtype=float)) for x in xs]
      for y in ys
    ])

    plt.figure(figsize=plt.rcParams['figure.figsize'])

    if func.name == 'rosenbrock':
      levels = np.logspace(-1, 3, 18)
      cs = plt.contour(x, y, z, levels=levels, alpha=0.75)
    else:
      cs = plt.contour(x, y, z, levels=25, alpha=0.75)

    plt.clabel(cs, inline=True, fontsize=7)
    plt.plot(traj[:, 0], traj[:, 1], 'o--', markersize=3, linewidth=1.2)
    plt.scatter([traj[0, 0]], [traj[0, 1]], marker='o')
    plt.scatter([traj[-1, 0]], [traj[-1, 1]], marker='x', s=160)
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

  @staticmethod
  def quadratic_grid_experiment(optimizers: list[BaseOptimizer], ns, ks, eps: float, seeds=(0, 1, 2)):
    raw_rows = []

    for n in ns:
      for k in ks:
        for seed in seeds:
          func = make_quadratic(n, k, seed=seed, name=f'quadratic n={n}, k={k}')
          x0 = np.full(n, 4.0, dtype=float)

          for opt in optimizers:
            res = Calculation._run(opt, func, x0, eps)
            raw_rows.append(Calculation._row(func, opt, res, n=n, k=k, seed=seed))

    rows = Calculation._aggregate(raw_rows, ['function', 'optimizer', 'n', 'k'])
    Calculation.print_table(rows, 'quadratic grid experiment')
    Calculation.print_table(Calculation._optimizer_summary(rows), 'quadratic methods comparison')

    fixed_k = 10
    Calculation._plot_metric_series(
      [r for r in rows if r['k'] == fixed_k],
      x_key='n',
      x_label='n',
      title_prefix=f'metrics n (k={fixed_k})',
    )

    fixed_n = 10
    Calculation._plot_metric_series(
      [r for r in rows if r['n'] == fixed_n],
      x_key='k',
      x_label='k',
      title_prefix=f'metrics k (n={fixed_n})',
    )

    return rows

  @staticmethod
  def quadratic_start_points_experiment(
      func: BaseFunction,
      optimizers: list[BaseOptimizer],
      starts,
      eps: float,
      contour_range,
  ):
    rows = []

    for x0 in starts:
      for opt in optimizers:
        res = Calculation._run(opt, func, x0, eps)
        rows.append(Calculation._row(func, opt, res, start=Calculation._format_vec(x0)))

        Calculation._plot_contour_trajectory(
          func,
          res.trajectory,
          f'{opt.name}: {func.name}, x0={Calculation._format_vec(x0)}',
          contour_range[0],
          contour_range[1],
        )

    Calculation.print_table(rows, '2d quadratic from different starting points')
    return rows

  @staticmethod
  def complex_functions_experiment(
      functions: list[BaseFunction],
      optimizers: list[BaseOptimizer],
      starts: dict,
      eps: float,
      contour_ranges: dict,
  ):
    rows = []

    for func in functions:
      func_rows = []
      x_range, y_range = contour_ranges[func.name]

      for x0 in starts[func.name]:
        for opt in optimizers:
          res = Calculation._run(opt, func, x0, eps)
          row = Calculation._row(func, opt, res, start=Calculation._format_vec(x0))
          rows.append(row)
          func_rows.append(row)

          Calculation._plot_contour_trajectory(
            func,
            res.trajectory,
            f'{opt.name}: {func.name}, x0={Calculation._format_vec(x0)}',
            x_range,
            y_range,
          )

      Calculation.print_table(func_rows, f'{func.name} results')

    return rows

  @staticmethod
  def lbfgs_memory_experiment(
      func: BaseFunction,
      x0: np.ndarray,
      ms,
      eps: float,
  ):
    rows = []

    for m in ms:
      opt = LBFGS(m=m)
      opt.name = opt.name.lower()
      res = Calculation._run(opt, func, x0, eps)
      rows.append(Calculation._row(func, opt, res, m=int(m)))

    Calculation.print_table(rows, 'l-bfgs memory experiment')

    plt.figure(figsize=plt.rcParams['figure.figsize'])
    for metric_key, metric_name in [('it', 'iterations'), ('f_calls', 'function calls'), ('g_calls', 'gradient calls')]:
      plt.plot([r['m'] for r in rows], [r[metric_key] for r in rows], marker='o', label=metric_name)

    plt.title('l-bfgs: influence of memory m')
    plt.xlabel('m')
    plt.ylabel('count')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return rows
