"""
06 — SAT instance as wave interference.

Each clause becomes a sum of plane waves at variable-specific wavenumbers.
The initial wavefunction Psi_0 is the superposition of all clause waves,
which encodes the constraint structure as an interference pattern.

Evolution under P1+P2+P3 lets the field explore the constraint landscape.
The structure it arrives at carries information about satisfying configurations.
"""
import numpy as np
from triad_pde.sat_solver import SATInstance, encode_sat_interference
from triad_pde.core.solver import TriadParams, integrate
from triad_pde.physics.observables import crystallinity, dominant_wavenumber

sat = SATInstance(
    n_vars=3,
    clauses=[
        [1, 2, -3],
        [-1, 2, 3],
        [1, -2, 3],
        [-1, -2, -3],
    ]
)

print(f"SAT instance: {sat.n_vars} variables, {len(sat.clauses)} clauses")
solutions = sat.all_solutions()
print(f"Solutions (brute force): {len(solutions)} of {2**sat.n_vars} assignments")
print()

encoded = encode_sat_interference(sat, N_grid=64, L=32.0)
psi0 = encoded['psi']
N_grid = encoded['N_grid']
L = encoded['L']
dx = L / N_grid

print(f"Encoded as interference pattern: N={N_grid}, L={L}")
print(f"  Initial crystallinity: {crystallinity(psi0, dx):.4f}")
print()

p = TriadParams(N=N_grid, L=L, T=5.0, dt=0.005, seed=0,
                V_ext=encoded['V_ext'] if callable(encoded['V_ext']) else None)
result = integrate(p, psi0=psi0)
psi_f = result['psi_final']

print("After P1+P2+P3 evolution:")
print(f"  final crystallinity: {crystallinity(psi_f, dx):.4f}")
print(f"  final k_star:        {dominant_wavenumber(psi_f, dx):.4f}")
print()
print("The constraint structure shapes the initial interference;")
print("the equation evolves it through its own dynamics.")
