"""
03 — coupled substrates.

The MultiRuntime evolves several fields in lockstep, with explicit coupling
edges between them. Each substrate runs the full P1+P2+P3 equation. The
coupling adds a V_couple term to V_total but does not impose structure.

The ring topology is one of three options (CouplingEdge graph can be
arbitrary). Here we couple three substrates from three different regimes.
"""
import numpy as np
from triad_pde.core.multi_runtime import MultiRuntime, CouplingEdge, Segment
from triad_pde.regimes import resolve_regime
from triad_pde.physics.observables import crystallinity, dominant_wavenumber

rt = MultiRuntime(dt=0.005, record_every=4)

p_a = resolve_regime('HodgkinHuxley', seed=0)
p_b = resolve_regime('ENSO_recharge',  seed=0)
p_c = resolve_regime('LSV_market',     seed=0)

sub_a = rt.add_substrate('neuron',  p_a)
sub_b = rt.add_substrate('climate', p_b)
sub_c = rt.add_substrate('market',  p_c)

kappa = -2.5
T = 12.0
ids = [sub_a.id, sub_b.id, sub_c.id]
edges  = [CouplingEdge(src_id=ids[i], dst_id=ids[(i+1)%3], kappa=kappa) for i in range(3)]
edges += [CouplingEdge(src_id=ids[(i+1)%3], dst_id=ids[i], kappa=kappa) for i in range(3)]

rt.add_segment(Segment(t_start=0.0, t_end=T, edges=edges))
rt.global_t = T

print(f"Ring coupling: 3 substrates, kappa={kappa}, T={T}")
print(f"Each substrate runs full P1+P2+P3 with its own regime parameters")
print()

import time
t0 = time.perf_counter()
rt.run(verbose=False)
elapsed = time.perf_counter() - t0

print(f"Integration finished in {elapsed:.2f}s")
print()
print(f"{'substrate':<12s}  {'regime':<20s}  {'cryst':>7s}  {'k*':>8s}")
print("-" * 56)
for sub, regime in zip([sub_a, sub_b, sub_c],
                       ['HodgkinHuxley', 'ENSO_recharge', 'LSV_market']):
    c = crystallinity(sub.psi, sub.dx)
    k = dominant_wavenumber(sub.psi, sub.dx)
    print(f"{sub.name:<12s}  {regime:<20s}  {c:>7.4f}  {k:>8.4f}")

print()
print("Each substrate settles its own structure. The ring propagates")
print("influence between them but does not impose a shared target.")
