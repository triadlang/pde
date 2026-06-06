"""
07 — real benchmark: FFT-based PDE vs naive Python DFT.

The only meaningful speed comparison for this solver is FFT vs the O(N^2)
DFT it replaces. Scalar loops like Fibonacci tell you nothing — the
equation's cost is dominated by the spectral half-step.

This script integrates the same physical problem two ways: once with the
real solver (NumPy FFT, O(N log N)), once with a pure Python implementation
that computes the DFT by definition (O(N^2)). Both use the full P1+P2+P3
update so they are doing the same physics, just with different FFT cost.
"""
import time
import math
import cmath
import numpy as np
from triad_pde.core.solver import TriadParams, integrate
from triad_pde.physics.observables import crystallinity


def pure_python_pde_step(N, L, T, dt, Lambda, Gamma):
    dx = L / N
    steps = int(T / dt)
    hbar = 1.0
    m = 1.0
    x = [-L/2 + i*dx for i in range(N)]
    k = [2 * math.pi * (i if i < N//2 else i - N) / L for i in range(N)]

    psi = [complex(math.exp(-xi**2 / 8.0), 0.0) for xi in x]
    norm = math.sqrt(sum(abs(p)**2 * dx for p in psi))
    psi = [p / norm for p in psi]

    def dft(arr):
        n = len(arr)
        out = [0j] * n
        for kk in range(n):
            s = 0j
            for j in range(n):
                s += arr[j] * cmath.exp(-2j * math.pi * kk * j / n)
            out[kk] = s
        return out

    def idft(arr):
        n = len(arr)
        out = [0j] * n
        for j in range(n):
            s = 0j
            for kk in range(n):
                s += arr[kk] * cmath.exp(2j * math.pi * kk * j / n)
            out[j] = s / n
        return out

    half_lin = [cmath.exp(-1j * (hbar**2 * ki**2 / (2*m)) * dt / (2*hbar)
                          - Gamma * dt / (2*hbar)) for ki in k]

    start = time.perf_counter()
    for step in range(steps):
        psi_k = dft(psi)
        psi_k = [psi_k[i] * half_lin[i] for i in range(N)]
        psi = idft(psi_k)
        for i in range(N):
            rho = abs(psi[i])**2
            V = Lambda * rho
            psi[i] *= cmath.exp(-1j * V * dt / hbar)
        psi_k = dft(psi)
        psi_k = [psi_k[i] * half_lin[i] for i in range(N)]
        psi = idft(psi_k)
    elapsed = time.perf_counter() - start
    return elapsed


N = 32
T = 0.5
dt = 0.005
steps = int(T / dt)

print(f"Comparison: solving the same equation two ways")
print(f"  N={N}, T={T}, dt={dt}, steps={steps}")
print()

print(f"Pure Python with naive DFT (O(N^2) per FFT = {N**2} ops/step):")
t_py = pure_python_pde_step(N=N, L=32.0, T=T, dt=dt, Lambda=4.0, Gamma=0.02)
print(f"  time: {t_py:.4f}s")
print()

print(f"triad_pde NumPy FFT solver (O(N log N) per FFT = ~{int(N*np.log2(N))} ops/step):")
p = TriadParams(N=N, L=32.0, T=T, dt=dt, seed=0, Lambda=4.0, Gamma=0.02)
t0 = time.perf_counter()
result = integrate(p, auto_halve_dt=False)
t_fft = time.perf_counter() - t0
print(f"  time: {t_fft:.4f}s")
print()

speedup = t_py / t_fft
print(f"Speedup: {speedup:.1f}x  (FFT vs DFT loop, same physics)")
print()
print(f"At N=128 the gap widens further:")
print(f"  DFT  cost: O(N^2)     = {128**2}")
print(f"  FFT  cost: O(N log N) = {int(128 * np.log2(128))}")
print(f"  ratio: {128**2 / (128 * np.log2(128)):.1f}x at one step")
print()
print(f"Final crystallinity: {crystallinity(result['psi_final'], result['dx']):.4f}")
print("The structure that emerged came from the equation, not from any optimization.")
