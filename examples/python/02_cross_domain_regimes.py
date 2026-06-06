"""
02 — cross-domain regimes.

Seventeen pre-defined parameter sets for the same equation, each producing
a different emergent phenomenology. The names describe what comes out of
the field, not what was put in.

No regime calibrates against a target. Each is a TriadParams configuration
that runs through the same P1+P2+P3 solver.
"""
import numpy as np
from triad_pde.core.solver import integrate
from triad_pde.regimes import resolve_regime, list_regimes
from triad_pde.physics.observables import crystallinity, dominant_wavenumber, ipr

print(f"Available regimes ({len(list_regimes())}):")
for r in list_regimes():
    print(f"  {r}")
print()

results = {}
for name in ['B0', 'R5_crystal', 'anti_collapse', 'HodgkinHuxley',
             'LSV_market', 'Cosmological_inflation', 'DarkMatter_halo',
             'ENSO_recharge', 'Belousov_Zhabotinsky']:
    p = resolve_regime(name, seed=0)
    r = integrate(p)
    psi = r['psi_final']
    dx = r['dx']
    results[name] = {
        'crystallinity': float(crystallinity(psi, dx)),
        'k_star': float(dominant_wavenumber(psi, dx)),
        'ipr': float(ipr(psi, dx)),
        'Lambda': p.Lambda,
        'sigma': p.sigma,
        'Gamma': p.Gamma,
    }

print(f"{'regime':<25s}  {'cryst':>7s}  {'k*':>8s}  {'IPR':>8s}  {'Lambda':>7s}  {'sigma':>5s}  {'Gamma':>5s}")
print("-" * 80)
for name, r in results.items():
    print(f"{name:<25s}  {r['crystallinity']:>7.4f}  {r['k_star']:>8.4f}  "
          f"{r['ipr']:>8.4f}  {r['Lambda']:>7.2f}  {r['sigma']:>5.2f}  {r['Gamma']:>5.3f}")

print()
print("Same equation. Different parameter triples. Different emergent structure.")
print("Note sigma is strictly below 2 in every regime — the fractional Laplacian")
print("is always non-trivial. P1+P2+P3 stay active in every case.")
