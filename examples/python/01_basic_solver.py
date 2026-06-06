"""
01 — basic solver: chaos to stabilization.

The Triad equation is integrated with the full split-step Strang scheme.
At t=0 the field is a Gaussian; under P1+P2+P3 it crystallizes on its own.
Nothing is imposed — the structure that emerges is whatever the equation
arrives at given these parameters.
"""
import numpy as np
from triad_pde.core.solver import TriadParams, integrate
from triad_pde.physics.observables import crystallinity, dominant_wavenumber, ipr

print("Triad equation: i·hbar·d_t Psi = [-hbar^2/(2m) nabla^2 + V_ext")
print("                + Lambda|Psi|^2 + V_mem + alpha(-Delta)^(sigma/2) - i*Gamma] Psi + eta")
print()

p = TriadParams(N=128, L=32.0, T=10.0, dt=0.005, seed=42,
                Lambda=-0.5, alpha=0.15, sigma=1.5,
                Gamma=0.05, f_FDT=0.002,
                nu=(2.0, 0.5, 0.1), lam=(-0.3, -0.2, -0.1))

print(f"Grid:        N={p.N}, L={p.L}, dx={p.L/p.N:.4f}")
print(f"Time:        T={p.T}, dt={p.dt}, steps={int(p.T/p.dt)}")
print(f"P1 (kinetic):  alpha={p.alpha}, sigma={p.sigma}  (fractional, sigma<2)")
print(f"P2 (memory):   Lambda={p.Lambda}, M={len(p.nu)} memory modes")
print(f"P3 (open):     Gamma={p.Gamma}, f_FDT={p.f_FDT}  (FDT-locked)")
print()

result = integrate(p)
psi = result['psi_final']
dx = result['dx']

print("Observables read from the final field state:")
print(f"  crystallinity = {crystallinity(psi, dx):.4f}")
print(f"  k_star        = {dominant_wavenumber(psi, dx):.4f}")
print(f"  IPR           = {ipr(psi, dx):.6f}")
print(f"  peak density  = {(np.abs(psi)**2).max():.4f}")
print()

density_traj = result['density']
print(f"Density trajectory shape: {density_traj.shape}  (N, n_snapshots)")
print(f"Initial peak density: {density_traj[:, 0].max():.4f}")
print(f"Final peak density:   {density_traj[:, -1].max():.4f}")
print()
print("Structure crystallizes from the equation itself. Not imposed.")
