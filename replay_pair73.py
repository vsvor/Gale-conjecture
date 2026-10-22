#!/usr/bin/env python3
"""Reproduce the useful paired split found for the supplied 73-point input.

This replay is intentionally specific to the original attachment. It checks
its SHA-256 fingerprint before applying index-based moves. No input is modified.
Place next to borsuk_search.py, tune_solution.py and split_tune.py.
"""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import numpy as np
import borsuk_search as bs
import tune_solution as ts
import split_tune as st

INPUT_SHA256 = '819e91d5440d8030ca390a54302f2ad1c277f95d083cacc2d93f627e67a28f85'

def exact_binary_geometry(x: np.ndarray) -> ts.Geometry:
    fractions = [[Fraction(float(v)) for v in row] for row in x]
    digits = max(v.denominator.bit_length()-1 for row in fractions for v in row)
    return ts.Geometry([[int(v*10**digits) for v in row] for row in fractions], digits)

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--backend', default='native', choices=['native','python','pysat','external','auto'])
    p.add_argument('--solver', default='kissat')
    p.add_argument('--sat-timeout', type=float, default=10.)
    a = p.parse_args()
    pool = None
    try:
        if a.out.exists():
            raise ValueError('Use a new output directory.')
        if hashlib.sha256(a.input.read_bytes()).hexdigest() != INPUT_SHA256:
            raise ValueError('This index-based replay requires the original 73-point attachment; fingerprint differs.')
        x = np.load(a.input, allow_pickle=False)
        if x.shape != (73, 3) or not np.isfinite(x).all():
            raise ValueError('Expected 73 finite three-dimensional points.')
        args = ts.build_parser().parse_args([str(a.input), '--backend', a.backend,
                    '--solver', a.solver, '--sat-timeout', str(a.sat_timeout)])
        args.slsqp_iters = 350
        args.full_constraints = False  # All omitted diameter constraints are checked.
        pool = ts.OraclePool(4, args)
        original = ts.assess(exact_binary_geometry(x), pool)
        if not original.exact:
            raise ValueError('Input evaluation unresolved; increase --sat-timeout.')
        keep = np.array([i for i in range(73) if i not in (67, 68, 69)])
        g = original.g.subset(keep)
        reduced = ts.assess(g, pool)
        if not reduced.exact:
            raise ValueError('Reduced input evaluation unresolved; increase --sat-timeout.')
        # This preprocessing can lose <1e-15. Acceptance below still compares
        # against the ORIGINAL 73-point upper bound, with exact arithmetic.
        ids = g.threshold(reduced.lower2)
        x0 = bs.normalize(g.x)
        w1 = st.split_edge_ids(70, g.pairs[ids], 26, (45,), (64,))
        y = np.vstack((x0, x0[26]))
        w2 = st.split_edge_ids(71, bs.pairs_for(71)[w1], 55, (26,), (15, 56))
        z = np.vstack((y, y[55]))
        for size, witness in ((71, w1), (72, w2)):
            status = pool.get(size).solve(witness).status
            if status is not bs.Status.UNSAT:
                raise ValueError(f'{size}-vertex scaffold not established UNSAT: {status.value}')
        first_audit = ts.linearized_audit(y, w1)[0]
        pair_audit = ts.linearized_audit(z, w2)[0]
        output, report = ts.slsqp_once(z, w2, args, args.ftol)
        result = ts.assess(ts.Geometry.from_float(bs.normalize(output)), pool, w2, witness_known=True)
        if result.lower2 <= original.upper2:
            raise ValueError('This numerical replay did not produce a certified improvement; input remains untouched.')
        meta = {'input_sha256': INPUT_SHA256, 'removed_original_indices': [67,68,69],
                'original_lower': original.lower, 'original_upper': original.upper,
                'first_split': {'vertex':26,'left':[45],'right':[64]},
                'second_split': {'vertex':55,'left':[26],'right':[15,56]},
                'first_audit':first_audit,'pair_audit':pair_audit,'slsqp':report,
                'qualification':'Auxiliary seeds may coincide; published output has distinct exact decimal rows. No optimality claim.'}
        ts.save_state(a.out, result, 4, meta)
        np.save(a.out/'optimization_seed.npy', z)
        np.savetxt(a.out/'optimization_edges.csv', bs.pairs_for(72)[w2], fmt='%d', delimiter=',')
        print(json.dumps({'input_ratio': original.lower, 'points': result.g.m,
                          'lower': result.lower, 'upper': result.upper, 'exact_squared':result.exact,
                          'output':str(a.out)}, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f'Error: {exc}')
        return 2
    finally:
        if pool is not None:
            pool.close()

if __name__ == '__main__':
    raise SystemExit(main())
