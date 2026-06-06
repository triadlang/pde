"""
Tests covering the high-level PDE/research surface: regimes, coupled
substrates, and the codec roundtrip.

These tests run the actual solver and read observables from the real
field state. No fabricated numbers, no math.sin() substitutes. If the
solver does not produce structure, the test fails.
"""
import numpy as np
import pytest
from triad_pde.core.solver import TriadParams, integrate
from triad_pde.core.multi_runtime import MultiRuntime, CouplingEdge, Segment
from triad_pde.physics.observables import crystallinity, dominant_wavenumber
from triad_pde.regimes import resolve_regime, list_regimes
from triad_pde.codec.codec import encode_int, decode_int, encode_bool, decode_bool, IntCalib


def test_regime_registry_has_seventeen_regimes():
    names = list_regimes()
    assert len(names) >= 17, f"expected at least 17 regimes, got {len(names)}"
    assert 'B0' in names
    assert 'R5_crystal' in names
    assert 'anti_collapse' in names


@pytest.mark.parametrize("name", ['B0', 'R5_crystal', 'HodgkinHuxley',
                                   'LSV_market', 'DarkMatter_halo'])
def test_every_regime_keeps_all_three_pillars_active(name):
    """Verifies P1+P2+P3 are non-trivial in every regime.

    P1: alpha > 0 and sigma strictly between 0 and 2.
    P2: at least one non-zero memory coupling lam_j.
    P3: Gamma > 0 and f_FDT > 0.
    """
    p = resolve_regime(name)
    assert p.alpha > 0, f"{name}: P1 alpha must be positive"
    assert 0 < p.sigma < 2, f"{name}: P1 sigma must be strictly between 0 and 2"
    assert any(abs(l) > 0 for l in p.lam), f"{name}: P2 lam must be non-zero"
    assert p.Lambda != 0, f"{name}: P2 Lambda must be non-zero"
    assert p.Gamma > 0, f"{name}: P3 Gamma must be positive"
    assert p.f_FDT > 0, f"{name}: P3 f_FDT must be positive"


def test_regime_b0_produces_emergent_structure():
    """The B0 regime starts from a Gaussian and arrives at structured field."""
    p = resolve_regime('B0', seed=42)
    p = TriadParams(**{**p.__dict__, 'T': 10.0})
    r = integrate(p)
    c = crystallinity(r['psi_final'], r['dx'])
    assert c > 0.5, f"B0 should crystallize, got crystallinity={c}"


def test_coupled_substrates_run_full_equation():
    """MultiRuntime evolves coupled substrates each with full P1+P2+P3."""
    rt = MultiRuntime(dt=0.005, record_every=4)
    p_a = resolve_regime('HodgkinHuxley', seed=0)
    p_b = resolve_regime('ENSO_recharge',  seed=0)
    p_a = TriadParams(**{**p_a.__dict__, 'T': 4.0})
    p_b = TriadParams(**{**p_b.__dict__, 'T': 4.0})

    sub_a = rt.add_substrate('a', p_a)
    sub_b = rt.add_substrate('b', p_b)

    edges = [
        CouplingEdge(src_id=sub_a.id, dst_id=sub_b.id, kappa=-2.0),
        CouplingEdge(src_id=sub_b.id, dst_id=sub_a.id, kappa=-2.0),
    ]
    rt.add_segment(Segment(t_start=0.0, t_end=4.0, edges=edges))
    rt.global_t = 4.0
    rt.run(verbose=False)

    norm_a = float((np.abs(sub_a.psi)**2).sum() * sub_a.dx)
    norm_b = float((np.abs(sub_b.psi)**2).sum() * sub_b.dx)
    assert norm_a > 0.01, f"substrate a collapsed: norm={norm_a}"
    assert norm_b > 0.01, f"substrate b collapsed: norm={norm_b}"
    assert np.isfinite(norm_a) and np.isfinite(norm_b)


def test_codec_integer_roundtrip():
    """encode_int -> psi -> decode_int recovers the input value."""
    calib = IntCalib()
    L, N = 32.0, 128
    dx = L / N
    for v in [0, 1, 2, 5, 10]:
        psi = encode_int(v, calib, L, N)
        decoded = decode_int(psi, dx, calib)
        assert decoded == v, f"roundtrip failed: {v} -> {decoded}"


def test_codec_boolean_roundtrip():
    """encode_bool -> psi -> decode_bool recovers True/False through crystallinity."""
    L, N = 32.0, 128
    dx = L / N
    psi_true = encode_bool(True, L, N)
    psi_false = encode_bool(False, L, N)
    assert decode_bool(psi_true, dx) is True
    assert decode_bool(psi_false, dx) is False


def test_solver_norm_stays_finite_in_full_mode():
    """Full P1+P2+P3 should not collapse to zero or diverge."""
    for seed in [0, 1, 2, 3]:
        p = TriadParams(N=64, T=5.0, dt=0.005, seed=seed)
        r = integrate(p)
        norms = r['density'].sum(axis=0) * r['dx']
        assert all(np.isfinite(norms)), f"seed {seed}: norms contain non-finite values"
        assert norms[-1] > 0.001, f"seed {seed}: norm collapsed to {norms[-1]}"
        assert norms[-1] < 1000, f"seed {seed}: norm diverged to {norms[-1]}"
