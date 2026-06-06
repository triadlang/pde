"""
04 — Gauss-Newton self-calibrator.

The calibrator perturbs {Lambda, Gamma, alpha, sigma, kT, f_FDT} and
measures the Jacobian of the field's emergent observables. The residual
it minimizes is between each observable and its own running median, not
between observables and any external target.

This is the principle: the field's own dynamics define what it converges
toward. The calibrator does not impose a label.
"""
import numpy as np
from triad_pde.core.solver import TriadParams
from triad_pde.calibrator import GaussNewtonCalibrator, CalibrationConfig

p = TriadParams(N=64, T=6.0, dt=0.005, seed=0)

cfg = CalibrationConfig(
    observables=['crystallinity', 'dominant_k'],
    max_iter=8,
    damping=1e-2,
    perturbation_frac=1e-3,
    integrate_T=6.0,
    verbose=False,
)

print("Self-calibration: residual is observable - running_median(observable).")
print("No external target. The field's own statistics define convergence.")
print()
print(f"Tunable parameters: {cfg.observables}")
print(f"Max iterations: {cfg.max_iter}")
print()

cal = GaussNewtonCalibrator(cfg)
result = cal.calibrate(p)

print(f"converged: {result['converged']}")
print(f"iterations: {result['n_iters']}")
print()
print("Final observables (read from the field, not predicted):")
for name, val in result['final_observables'].items():
    print(f"  {name}: {val:.6f}")

print()
print("Final parameters (after calibration):")
final_p = result['final_params']
for name in ['Lambda', 'Gamma', 'alpha', 'sigma', 'kT', 'f_FDT']:
    if name in final_p:
        print(f"  {name}: {final_p[name]:.6f}")

print()
print("Not converging in N steps is consistent with the principle —")
print("the field's observables are still in their stabilization phase.")
