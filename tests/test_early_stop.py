"""Early-stop policy, worker lifecycle, and real multiprocessing regression tests.

Run: python -m unittest -v test_borsuk_search test_early_stop
"""
from concurrent.futures import ProcessPoolExecutor
import contextlib
import io
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import borsuk_search as b

ROOT = Path(b.__file__).resolve().parent
_READY = _RELEASE = None


def init_live_test(record, ready, release):
    global _READY, _RELEASE
    b._init_search_worker(record)
    _READY, _RELEASE = ready, release


def live_leader():
    b._publish_search_record(0.91)
    _READY.set()
    if not _RELEASE.wait(20):
        raise RuntimeError('Follower did not observe the live record.')
    return b._search_record_lower()


def live_follower():
    if not _READY.wait(20):
        raise RuntimeError('Leader did not publish.')
    seen = b._search_record_lower()
    b._publish_search_record(0.80)  # An older/weaker update must not lower it.
    status = b.early_stop_status(
        {'early_stop': [(2, .02)], 'early_stop_warmup': 0,
         'early_stop_keep_every': 0}, 1, 2, .85, seen)
    after = b._search_record_lower()
    _RELEASE.set()
    return seen, after, status


class PolicyTests(unittest.TestCase):
    def config(self, **kwargs):
        cfg = {'early_stop': [(2, .02), (5, .005)],
               'early_stop_warmup': 0, 'early_stop_keep_every': 0}
        return dict(cfg, **kwargs)

    def test_parse_schedule(self):
        self.assertEqual(b.parse_early_stop('2:0.02, 5:0.005'), [(2, .02), (5, .005)])
        self.assertEqual(b.parse_early_stop('3:0'), [(3, 0.)])

    def test_invalid_schedules(self):
        for text in ('', '3', '0:.1', '-1:.1', '2:-.1', '2:nan', '2:inf',
                     '2:1.1', '3:.1,2:.01', '2:.1,2:.01', '2.5:.1', '2:.1,'):
            with self.subTest(text=text), self.assertRaises(b.argparse.ArgumentTypeError):
                b.parse_early_stop(text)

    def test_disabled(self):
        self.assertFalse(b.early_stop_status({}, 2, 10, .2, .9)['stop'])

    def test_initial_evaluation_is_not_round_one(self):
        self.assertEqual(b.early_stop_status(self.config(), 1, 1, .2, .9)['reason'],
                         'before_first_check')

    def test_first_checkpoint(self):
        self.assertTrue(b.early_stop_status(self.config(), 1, 2, .87, .9)['stop'])

    def test_rule_persists_between_checkpoints(self):
        self.assertTrue(b.early_stop_status(self.config(), 1, 4, .87, .9)['stop'])

    def test_tighter_checkpoint(self):
        self.assertFalse(b.early_stop_status(self.config(), 1, 4, .885, .9)['stop'])
        self.assertTrue(b.early_stop_status(self.config(), 1, 5, .885, .9)['stop'])

    def test_last_rule_remains_active(self):
        self.assertEqual(b.early_stop_status(self.config(), 1, 50, .885, .9)['rule_round'], 5)

    def test_equality_and_roundoff_do_not_stop(self):
        for u in (.88, .88 - 5e-13):
            self.assertFalse(b.early_stop_status(self.config(), 1, 2, u, .9)['stop'])

    def test_warmup_and_exploration_use_numbered_restarts(self):
        cfg = self.config(early_stop_warmup=2, early_stop_keep_every=10)
        self.assertEqual(b.early_stop_status(cfg, 0, 8, .2, .9)['reason'], 'protected_warmup')
        self.assertEqual(b.early_stop_status(cfg, 1, 8, .2, .9)['reason'], 'protected_warmup')
        self.assertEqual(b.early_stop_status(cfg, 9, 8, .2, .9)['reason'], 'protected_exploration')
        self.assertEqual(b.early_stop_status(cfg, 19, 8, .2, .9)['reason'], 'protected_exploration')
        self.assertTrue(b.early_stop_status(cfg, 2, 8, .2, .9)['stop'])

    def test_no_record_or_unresolved_upper(self):
        for record in (0., float('nan')):
            self.assertFalse(b.early_stop_status(self.config(), 1, 5, .2, record)['stop'])
        for upper in (1., float('inf'), float('nan')):
            self.assertFalse(b.early_stop_status(self.config(), 1, 5, upper, .9)['stop'])

    def test_record_holder_is_not_culled(self):
        self.assertFalse(b.early_stop_status(self.config(), 1, 10, .91, .91)['stop'])


class RestartFlowTests(unittest.TestCase):
    """Instrument expensive routines to isolate the control-flow decisions."""
    def run_case(self, schedule, values, *, rounds=8, warmup=0, keep=0, run=1,
                 record=.9, polish=10):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(dim=3, k=4, m_min=5, m_max=5, init='sphere', sphere=False,
                       start=None, mutation=.02, seed_jitter=.01, out=tmp,
                       backend='python', solver='kissat', sat_timeout=1., pysat_solver='g4',
                       quiet=True, rounds=rounds, graph_slack=.025, prune_checks=0,
                       steps=1, polish=polish, early_stop=schedule,
                       early_stop_warmup=warmup, early_stop_keep_every=keep)
            seen = iter(values)
            last = values[-1]
            def evaluate(x, oracle):
                lo, hi = next(seen, last)
                return b.Evaluation(lo, hi, lo == hi, np.arange(10, dtype=np.int32),
                                    np.array([0, 1, 2, 3, 0]), 0, 10, lo != hi)
            def save(directory, x, k, ev, pairs, metadata):
                directory.mkdir(parents=True, exist_ok=True)
                (directory / 'points.npy').write_bytes(b'control-flow test fixture')
                return {'geometry': {'lower_decimal': str(ev.lower), 'upper_decimal': str(ev.upper)}}
            shared = mp.get_context().Value('d', record)
            previous = b._SEARCH_RECORD
            b._init_search_worker(shared)
            try:
                with patch.object(b, 'evaluate', side_effect=evaluate), \
                     patch.object(b, 'save_candidate', side_effect=save), \
                     patch.object(b, 'prune_witness', side_effect=lambda o, ids, *a, **kw: ids), \
                     patch.object(b, 'optimize_witness', side_effect=lambda x, *a, **kw: (x.copy(), 0.)) as opt, \
                     patch.object(b, 'replenish_unused', side_effect=lambda x, *a, **kw: x), \
                     patch.object(b, 'mutate', side_effect=lambda x, *a, **kw: x) as mut, \
                     patch.object(b, 'slsqp_polish', side_effect=lambda x, *a, **kw: (x, 0.)) as pol:
                    result = b._run_restart(cfg, run, 123)
                    status_file = Path(tmp)/f'run_{run:05d}'/'status.json'
                    self.assertEqual(json.loads(status_file.read_text()), result)
                    self.assertTrue((Path(result['path'])/'points.npy').exists())
                    return result, opt.call_count, mut.call_count, pol.call_count, shared.value
            finally:
                b._init_search_worker(previous)

    def test_stop_after_two_and_skip_mutation_and_polish(self):
        r, opt, mut, pol, _ = self.run_case([(2, .02)], [(.8, .8)])
        self.assertTrue(r['early_stopped'])
        self.assertEqual((r['rounds_completed'], r['rounds_skipped'], opt, mut, pol), (2, 6, 2, 0, 0))
        self.assertTrue(r['polish_skipped_by_early_stop'])

    def test_second_checkpoint_stops_after_five(self):
        r, opt, _, pol, _ = self.run_case([(2, .02), (5, .005)], [(.885, .885)])
        self.assertEqual((r['rounds_completed'], opt, pol), (5, 5, 0))

    def test_historical_best_survives_worse_mutation(self):
        r, opt, _, pol, _ = self.run_case([(2, .02)], [(.895, .895), (.8, .8)])
        self.assertFalse(r['early_stopped'])
        self.assertEqual((opt, pol), (8, 1))
        self.assertEqual(r['best_seen_upper'], .895)

    def test_unsaved_timeout_candidate_protects_restart(self):
        r, opt, _, pol, _ = self.run_case([(2, .02)], [(.8, .8), (.2, .99), (.8, .8)])
        self.assertFalse(r['early_stopped'])
        self.assertEqual(r['best_seen_upper'], .99)
        self.assertEqual((opt, pol), (8, 1))

    def test_disabled_runs_all_rounds_and_polish(self):
        r, opt, _, pol, _ = self.run_case(None, [(.8, .8)])
        self.assertEqual((opt, pol, r['rounds_skipped']), (8, 1, 0))
        self.assertFalse(r['early_stopped'])

    def test_first_restart_protected(self):
        r, opt, _, pol, _ = self.run_case([(2, .02)], [(.8, .8)], warmup=1, run=0)
        self.assertEqual((opt, pol), (8, 1))
        self.assertFalse(r['early_stopped'])

    def test_final_round_can_skip_only_polish(self):
        r, opt, _, pol, _ = self.run_case([(2, .02)], [(.8, .8)], rounds=2)
        self.assertEqual((opt, pol, r['rounds_skipped']), (2, 0, 0))
        self.assertTrue(r['early_stopped'])

    def test_zero_rounds_still_polishes(self):
        r, opt, _, pol, _ = self.run_case([(2, .02)], [(.8, .8)], rounds=0)
        self.assertEqual((opt, pol), (0, 1))
        self.assertFalse(r['early_stopped'])

    def test_publication_updates_record_before_finish(self):
        r, _, _, _, record = self.run_case([(2, .02)], [(.93, .93)], record=.9, rounds=2)
        self.assertFalse(r['early_stopped'])
        self.assertAlmostEqual(record, .93)


class ProcessAndCLITests(unittest.TestCase):
    def test_live_sharing_with_all_available_start_methods(self):
        for method in mp.get_all_start_methods():
            with self.subTest(method=method):
                context = mp.get_context(method)
                record = context.Value('d', .5)
                ready, release = context.Event(), context.Event()
                with ProcessPoolExecutor(max_workers=2, mp_context=context,
                                         initializer=init_live_test,
                                         initargs=(record, ready, release)) as pool:
                    leader = pool.submit(live_leader)
                    follower = pool.submit(live_follower)
                    try:
                        seen, after, status = follower.result(timeout=35)
                        self.assertEqual((seen, after), (.91, .91))
                        self.assertTrue(status['stop'])
                        self.assertEqual(leader.result(timeout=10), .91)
                    finally:
                        release.set()

    def test_real_cli_under_spawn(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            launcher = tmp/'launch.py'
            launcher.write_text('import sys, multiprocessing as mp\n'
                                f'sys.path.insert(0, {str(ROOT)!r})\n'
                                'import borsuk_search as b\n'
                                'if __name__ == "__main__":\n'
                                '    mp.set_start_method("spawn")\n'
                                '    raise SystemExit(b.main())\n')
            out = tmp/'results'
            cmd = [sys.executable, str(launcher), 'search', '--dim', '3', '--points', '7',
                   '--rounds', '4', '--steps', '1', '--polish', '0', '--prune-checks', '0',
                   '--restarts', '6', '--workers', '2', '--backend', 'python', '--quiet',
                   '--early-stop', '1:0', '--early-stop-keep-every', '0', '--out', str(out)]
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            summary = json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['restarts'], 6)
            self.assertGreater(summary['early_stopped'], 0)
            self.assertEqual(summary['rounds_completed'] + summary['rounds_skipped'], 24)
            warmup = next(r for r in summary['results'] if r['run'] == 0)
            self.assertFalse(warmup['early_stopped'])
            self.assertEqual(warmup['rounds_completed'], 4)
            self.assertTrue(b.verify_candidate(out/'best', 'python', 'kissat', 5., 'g4')['verified'])

    def test_bare_flag_uses_default_schedule_and_main_restores_global(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = mp.get_context().Value('d', .7)
            original = b._SEARCH_RECORD
            b._init_search_worker(sentinel)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    code = b.main(['search', '--dim', '3', '--points', '4', '--restarts', '1',
                                   '--rounds', '3', '--steps', '1', '--polish', '0',
                                   '--backend', 'python', '--early-stop', '--quiet', '--out', tmp])
                self.assertEqual(code, 0)
                self.assertIs(b._SEARCH_RECORD, sentinel)
                cfg = json.loads((Path(tmp)/'config.json').read_text())
                self.assertEqual(cfg['early_stop'], [[2, .02], [5, .005]])
                self.assertEqual(cfg['early_stop_warmup'], 1)
                self.assertEqual(cfg['early_stop_keep_every'], 10)
            finally:
                b._init_search_worker(original)

    def test_negative_protection_counts_rejected(self):
        for option in ('--early-stop-warmup', '--early-stop-keep-every'):
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
                code = b.main(['search', '--dim', '3', option, '-1', '--out', tmp])
                self.assertEqual(code, 2)
                self.assertFalse((Path(tmp)/'config.json').exists())

    @unittest.skipUnless((ROOT/'dsatur_native.so').is_file(), 'Compile the optional native backend first.')
    def test_native_loader_with_relative_dunder_file(self):
        # Simulate launchers/older Python versions providing a relative __file__.
        import os
        cwd = Path.cwd()
        try:
            os.chdir(ROOT)
            with patch.object(b, '__file__', 'borsuk_search.py'):
                with b.GraphOracle(3, 2, backend='native') as oracle:
                    self.assertIs(oracle.solve([0, 1, 2]).status, b.Status.UNSAT)
        finally:
            os.chdir(cwd)


if __name__ == '__main__':
    unittest.main()
