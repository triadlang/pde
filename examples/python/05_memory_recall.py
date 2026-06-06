"""
05 — associative memory recall via P2 memory fields.

The memory fields y_j store density patterns over time through their
OU low-pass filtering. V_mem = sum(lam_j * y_j) acts as an attractor
landscape. Recall is the field relaxing toward the nearest stored
pattern under the full P1+P2+P3 dynamics. No lookup table. No imposed
target. The attractor structure comes from the equation.
"""
import numpy as np
from triad_pde.core.solver import TriadParams
from triad_pde.memory_recall import HopfieldMemory, overlap, capacity_estimate

N = 64
L = 32.0
dx = L / N

print(f"Hopfield-style capacity for N={N}: K ~ {capacity_estimate(N):.1f} patterns")
print("(Continuous-field scaling, not binary spin storage.)")
print()

x = np.linspace(-L/2, L/2, N, endpoint=False)

pat_left  = np.exp(-(x + 6)**2 / 2.0);  pat_left  /= pat_left.sum()  * dx
pat_right = np.exp(-(x - 6)**2 / 2.0);  pat_right /= pat_right.sum() * dx

print(f"Pattern 1 (left):  peak at x = {x[np.argmax(pat_left)]:.1f}")
print(f"Pattern 2 (right): peak at x = {x[np.argmax(pat_right)]:.1f}")
print(f"Overlap(left, right): {overlap(pat_left, pat_right):.4f}  (~0 = orthogonal)")
print()

p = TriadParams(N=N, L=L, T=4.0, seed=0)
hm = HopfieldMemory(params=p)

sp1 = hm.store_pattern(pat_left,  T_store=4.0)
sp2 = hm.store_pattern(pat_right, T_store=4.0)

print(f"Stored 2 patterns. y_stored shapes: {sp1.y_stored.shape}, {sp2.y_stored.shape}")
print()

rng = np.random.default_rng(0)
corrupted = pat_left.copy()
idx = rng.choice(N, size=N // 4, replace=False)
corrupted[idx] = rng.uniform(0, 0.01, N // 4)
corrupted = np.clip(corrupted, 0, None)

print(f"Cue: 25% of left pattern replaced with noise.")
print(f"  Overlap(cue, pat_left): {overlap(corrupted, pat_left):.4f}")
print()

result = hm.recall(corrupted, T_recall=4.0)
print(f"Recall result:")
print(f"  best_match index: {result['best_match']}  (0=left, 1=right)")
print(f"  best_overlap:     {result['best_overlap']:.4f}")
print(f"  overlaps:         {[f'{o:.4f}' for o in result['overlaps']]}")
print()
print("The field relaxes toward the stored attractor through V_mem.")
print("P1+P2+P3 all active during recall — dispersion explores, memory")
print("guides, FDT noise helps escape spurious basins.")
