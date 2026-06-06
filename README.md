# triad-pde

PDE solver and research toolkit for the Triad equation. Part of the [TriadLang](https://github.com/triadlang/triadlang) project, separated out so the equation, its solver, and the analysis tools that sit on top of it can be used without the language frontend, compiler, or CLI.

```
i·ℏ·∂_t Ψ = [-ℏ²/(2m)∇² + V_ext + Λ|Ψ|² + V_mem + α(-Δ)^(σ/2) - iΓ]Ψ + η
```

with auxiliary memory fields `∂_t y_j = ν_j (ρ - y_j)` and FDT-locked noise.

This is what runs the field. P1 (kinetic and fractional dispersion), P2 (cubic and memory), P3 (dissipation and noise) are always active together. Structure that appears in the field comes from the equation itself, not from any imposed target.

---

## What is here

### The core solver

```
src/triad_pde/core/
├── solver.py               Split-step Strang integrator (1D / 2D / 3D)
├── multi_runtime.py        Coupled substrates with CouplingEdge graph
├── fast_solver.py          Anderson-accelerated steady-state finder
├── equilibrium_solver.py   Fixed-point relaxation under boundary drive
├── batch_solver.py         K independent trajectories in parallel
├── equation_runtime.py     High-level multi-substrate driver
├── field_router.py         Precision-weighted MoE-style gating
└── stencil_ir.py           MLIR-style stencil dialect for vectorization
```

### Observables (read from the field, never imposed)

```
src/triad_pde/physics/
├── observables.py          Norm, peak, FWHM, IPR, k*, crystallinity,
│                           power spectrum, energy, FDT diagnostics
├── observables_atoms.py    Density-cluster counting (1D / 2D / 3D)
├── causal.py               Transfer entropy, memory causal strength
├── consciousness.py        Integrated information, Lempel-Ziv complexity
├── energy_readout.py       EBT-style relaxation, energy features
├── physical_reservoir.py   Edge-of-chaos reservoir emulator, Lyapunov
├── projections.py          Cross-substrate projection observables
└── qubits.py               Discrete qubit circuit interface
```

### Tools that sit on the equation

| Module | What it does |
|---|---|
| `regimes.py` | Seventeen `TriadParams` configurations producing different emergent phenomenologies. Every one has P1+P2+P3 active. Names describe what the field arrives at. |
| `calibrator.py` | Gauss-Newton self-calibration. Residual is observable - running_median(observable). No external target. |
| `memory_recall.py` | Hopfield-style associative memory through the P2 fields `y_j`. Recall is emergent attractor descent in V_mem. |
| `sat_solver.py` | SAT encoded as wave interference in the initial Psi_0. |
| `steering.py` | Bounded PID over V_ext only. Never clamps Psi. |
| `sigma_curriculum.py` | Fractional exponent sigma as a curriculum variable, adjusted from within P1. |
| `observers.py` | Passive read-only observers over recorded trajectories. Never feed back into the equation. |

### Codec

```
src/triad_pde/codec/
├── codec.py            Integer ↔ k*,  Bool ↔ crystallinity,  Float ↔ IPR
└── codec_3d.py         3D variants
```

The codec maps values to wavefunctions before the solver runs, and reads them back from observables after.

---

## Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[science]"   # adds scipy
pip install -e ".[viz]"        # adds matplotlib
pip install -e ".[gpu]"        # adds CuPy for CUDA backend
pip install -e ".[dev]"        # adds pytest
```

---

## A first run

```python
from triad_pde.core.solver import TriadParams, integrate
from triad_pde.physics.observables import crystallinity, dominant_wavenumber

p = TriadParams(
    N=128, L=32.0, T=10.0, dt=0.005, seed=42,
    Lambda=-0.5, alpha=0.15, sigma=1.5,    # P1+P2
    Gamma=0.05, f_FDT=0.002,                # P3 (FDT-locked)
    nu=(2.0, 0.5, 0.1),                     # P2 memory timescales
    lam=(-0.3, -0.2, -0.1),                 # P2 memory couplings
)

result = integrate(p)
psi = result['psi_final']
dx = result['dx']

print(f"crystallinity = {crystallinity(psi, dx):.4f}")
print(f"k_star        = {dominant_wavenumber(psi, dx):.4f}")
```

With these parameters at seed=42, the field arrives at `crystallinity ≈ 0.91, k* ≈ 6.87` from a Gaussian initial condition. Reproducible, deterministic given the seed.

---

## Coupled substrates

```python
from triad_pde.core.multi_runtime import MultiRuntime, CouplingEdge, Segment
from triad_pde.regimes import resolve_regime

rt = MultiRuntime(dt=0.005, record_every=4)

sub_a = rt.add_substrate('neuron',  resolve_regime('HodgkinHuxley'))
sub_b = rt.add_substrate('climate', resolve_regime('ENSO_recharge'))
sub_c = rt.add_substrate('market',  resolve_regime('LSV_market'))

ids = [sub_a.id, sub_b.id, sub_c.id]
edges  = [CouplingEdge(src_id=ids[i], dst_id=ids[(i+1)%3], kappa=-2.5)
          for i in range(3)]
edges += [CouplingEdge(src_id=ids[(i+1)%3], dst_id=ids[i], kappa=-2.5)
          for i in range(3)]

rt.add_segment(Segment(t_start=0.0, t_end=12.0, edges=edges))
rt.global_t = 12.0
rt.run(verbose=False)
```

Three substrates from three different regimes, coupled in a ring. Each runs the full equation with its own parameters. The coupling propagates influence but does not impose a target.

---

## The mathematical reference

`docs/triad_equation_reference.md` is a 1155-line technical specification: the equation, the numerical scheme, the controls used for verification, the observables, and the parameter regimes. Reproducible end-to-end from that document alone.

Sections 1–4 are the minimum to reproduce. Sections 5–13 cover extensions and operational details. Appendix A.3 documents the default parameter choices.

---

## Native C path

The C sources reimplement the solver, multi-runtime, observables, and codec with the same numerics as the Python side. Parity between Python and C is verified to fp64 precision in the parent project's parity suite.

| File | Lines | Role |
|---|---|---|
| `native/c/src/triad_solver.c` | 797 | 1D / 2D / 3D split-step Strang |
| `native/c/src/triad_multi_runtime.c` | 813 | Coupled substrates in C |
| `native/c/src/triad_observables.c` | 380 | Field measurements |
| `native/c/src/triad_observables_atoms.c` | 590 | Atom counting |
| `native/c/src/triad_codec.c` | 220 | Value ↔ wavefunction codec |
| `native/c/src/triad_fftw.c` | 240 | FFTW3 wrapper |
| `native/c/src/triad_blas.c` | 80 | BLAS helpers |

Building these requires the rest of the TriadLang native runtime (`triad_rt.h`, Boehm GC, FFTW3). The Python package works standalone.

---

## Examples

`examples/python/` — runnable Python scripts:

1. `01_basic_solver.py` — chaos at t=0, structure at t=T
2. `02_cross_domain_regimes.py` — nine of the seventeen regimes side by side
3. `03_coupled_substrates.py` — ring of three substrates from three regimes
4. `04_self_calibrator.py` — Gauss-Newton calibration with no external target
5. `05_memory_recall.py` — Hopfield recall through P2 memory fields
6. `06_sat_interference.py` — SAT encoded as initial interference pattern
7. `07_pde_vs_dft_benchmark.py` — FFT solver vs naive Python DFT, same physics

`examples/tri/` — original `.tri` programs from the parent project. These need the TriadLang language runtime to execute; the Python API above does not.

---

## Tests

```bash
pytest tests/
```

20 tests pass. These cover:

- Numerical audit: norm conservation to fp64 in the linear audit mode, dt convergence, FDT noise scaling, full-mode boundedness
- Regime registry: at least 17 regimes, every one keeps P1+P2+P3 active
- Multi-substrate: coupled run preserves finite norm in every substrate
- Codec: integer and boolean roundtrip
- Emergent structure: B0 crystallizes from a Gaussian

The audit tests use `mode='linear'` and `mode='thermal'` as verification harnesses only. The physical model is always `'full'`. No regime, configuration, or user-facing path can switch them off.

---

## What is not in this package

- The TriadLang language frontend (`lexer`, `parser`, `ast_nodes`)
- The compiler (`lower`, `c_codegen`, `triadc`)
- The CLI (`triad run`, `triad compile`, `triad repl`)
- The ML stack (`TriadSSM`, `TriadLM`, tensor autograd, training loop)
- The VM (`TriadVM` with opcodes)
- The interpreter, adapters, embed engine

For the language and the ML stack:
- main repo: [TriadLang](https://github.com/triadlang/triadlang)
- ML subset: [triadlang/machine-learning](https://github.com/triadlang/machine-learning)

This package is the PDE solver and the analysis tools that sit on it. That is the whole point of separating it: research code that just wants to integrate the equation and read observables does not need the rest.

---

## License

Same terms as the parent TriadLang project. See `LICENSE`.
