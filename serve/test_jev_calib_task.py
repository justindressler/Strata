"""serve/test_jev_calib_task.py - task-scoped calibration (cross-task contamination guard).

    python -m unittest serve.test_jev_calib_task -v

The defect this pins down (measured 2026-11-01, 500 banking77 items, single GPU):

  swap() keys fits by READOUT LETTER, so one calib-L.json served every text task.
  A fit carrying a_noul=0.6127, c_noul=+0.6539 (fitted on BoolQ60/ARC60/SST5-60)
  was applied to a balanced binary idea-grouping task that needs c_noul ~ -1.56.
  Effect: yes-rate 67.6% -> 80.4%, specificity 0.584 -> 0.360, accuracy
  0.760 -> 0.664 (McNemar exact p < 0.001).  Doing NOTHING beat what shipped.

Rules pinned here:
  1. no task tag in the environment -> legacy behaviour, byte-identical (no regression);
  2. a fit recorded for the same task -> applied (platt_v1);
  3. a fit recorded for a DIFFERENT task -> refused, identity, probabilities untouched;
  4. an UNTAGGED fit with a task tag set -> refused (fail closed: identity beats a
     stranger's intercept, and an untagged file cannot prove it belongs here);
  5. refusal is reported in the calib id so callers can see it, not silent.
"""
from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("STRATA_JEV_MODES", "chat,decision")
from serve import jev_calib  # noqa: E402


def _fit(task: str, a: float = 0.6127, c: float = 0.6539) -> "jev_calib.Calib":
    return jev_calib.Calib(loaded=True, tuple_str="qfn|jev-port|none|v1|L|text",
                           T_choice=1.0557, T_score=2.263,
                           a_noul=a, c_noul=c, task=task)


class TaskScopedCalibration(unittest.TestCase):
    def setUp(self):
        self._snapshot()
        self._env_task = os.environ.pop("STRATA_JEV_TASK", None)
        jev_calib._set_task_for_test(None)      # fall back to the environment
        with jev_calib._MU:
            jev_calib._ACTIVE.clear()
            jev_calib._ACTIVE[""] = None
        jev_calib._MISMATCH_LOGGED.clear()

    def tearDown(self):
        self._restore()

    # ── state isolation ────────────────────────────────────────────────────
    # _set_for_test(c) defaults to mode "", so resetting it alone leaves a fit
    # parked in _ACTIVE["L"].  That leak is not theoretical: it failed CI's
    # "Minimal serving runtime" job on 2026-10-31 (run 41246840308), three tests
    # in serve/test_systemone.py that assert uncalibrated probabilities.  Files
    # run alphabetically, so test_jev_calib_task finished first and its leftover
    # Calib(a_noul=0.6127, c_noul=+0.6539, T_choice=1.0557, T_score=2.263) was
    # applied by the server those tests start:
    #   noul  0.982   -> 0.9570665554  (CI observed 0.9570665554111856)
    #   score 0.9887011299 -> 0.9524334890 (CI observed exactly that)
    #   choice 0.0178982102 -> 0.0219656216 (CI observed exactly that)
    # Restore EVERY mode, the task override, and the log set.
    def _snapshot(self):
        with jev_calib._MU:
            self._active = dict(jev_calib._ACTIVE)
        self._task_override = jev_calib._TASK_OVERRIDE
        self._mismatch = set(jev_calib._MISMATCH_LOGGED)

    def _restore(self):
        with jev_calib._MU:
            jev_calib._ACTIVE.clear()
            jev_calib._ACTIVE.update(self._active)
        jev_calib._set_task_for_test(self._task_override)
        jev_calib._MISMATCH_LOGGED.clear()
        jev_calib._MISMATCH_LOGGED.update(self._mismatch)
        os.environ.pop("STRATA_JEV_TASK", None)
        if self._env_task is not None:
            os.environ["STRATA_JEV_TASK"] = self._env_task

    # ── 1. legacy path unchanged ───────────────────────────────────────────
    def test_untagged_runtime_applies_fit_as_before(self):
        fit = _fit("something-else")
        jev_calib._set_for_test(fit, "L")
        pr, cid = jev_calib.apply("noul", [0.5, 0.5], mode="L")
        self.assertEqual(cid, "platt_v1",
                         "no task tag must behave exactly as the last release")
        self.assertAlmostEqual(pr[1], 1.0 / (1.0 + math.exp(-0.6539)), places=6)

    # ── 2. matching task applies ───────────────────────────────────────────
    def test_matching_task_applies_fit(self):
        jev_calib._set_task_for_test("banking77-mf")
        jev_calib._set_for_test(_fit("banking77-mf", a=0.4504, c=-1.5607), "L")
        pr, cid = jev_calib.apply("noul", [0.99, 0.01], mode="L")
        self.assertEqual(cid, "platt_v1")
        z = math.log(0.01 / 0.99)
        self.assertAlmostEqual(pr[1], 1.0 / (1.0 + math.exp(-(0.4504 * z - 1.5607))), places=6)

    # ── 3. different task is refused ───────────────────────────────────────
    def test_other_task_fit_is_refused_and_reported(self):
        jev_calib._set_task_for_test("banking77-mf")
        jev_calib._set_for_test(_fit("boolq-arc-sst5"), "L")
        before = [0.99, 0.01]
        pr, cid = jev_calib.apply("noul", list(before), mode="L")
        self.assertEqual(cid, "identity", "a fit from another task must not be applied")
        self.assertEqual(pr, before, "probabilities must be untouched under refusal")

    # ── 4. untagged fit fails closed ───────────────────────────────────────
    def test_untagged_fit_is_refused_when_task_tag_set(self):
        jev_calib._set_task_for_test("banking77-mf")
        jev_calib._set_for_test(_fit(""), "L")
        pr, cid = jev_calib.apply("noul", [0.99, 0.01], mode="L")
        self.assertEqual(cid, "identity")
        self.assertEqual(pr, [0.99, 0.01])

    # ── 5. refusal also covers choice/score (not just noul) ────────────────
    def test_choice_temperature_also_scoped(self):
        jev_calib._set_task_for_test("banking77-mf")
        jev_calib._set_for_test(_fit("boolq-arc-sst5"), "L")
        pr, cid = jev_calib.apply("choice", [0.5, 0.3, 0.2], mode="L")
        self.assertEqual(cid, "identity")
        self.assertEqual(pr, [0.5, 0.3, 0.2])

    # ── 6. loader round-trips the task field ───────────────────────────────
    def test_loader_reads_task_field(self):
        import json, tempfile
        d = Path(tempfile.mkdtemp())
        (d / "calib-L.json").write_text(json.dumps({
            "tuple": "qfn|jev-port|none|v1|L|banking77-mf",
            "task": "banking77-mf",
            "a_noul": 0.4504, "c_noul": -1.5607,
            "T_choice": 1.0557, "T_score": 2.263,
            "provenance": "unit test"}))
        c = jev_calib.load(d / "calib-L.json")
        self.assertTrue(c.loaded)
        self.assertEqual(c.task, "banking77-mf")
        self.assertAlmostEqual(c.c_noul, -1.5607, places=6)

    # ── 7a. this file must not leak a fit into other test files ────────────
    def test_restore_clears_every_mode(self):
        jev_calib._set_for_test(_fit("leaky"), "L")          # what a test leaves behind
        jev_calib._set_for_test(_fit("leaky"), "")
        self.assertIsNotNone(jev_calib.active("L"))
        self._restore()                                      # what tearDown does
        self.assertIsNone(jev_calib.active("L"),
                          "a leftover fit in mode 'L' is applied by later test files")
        self.assertIsNone(jev_calib.active(""))
        # and the legacy fallback path in apply() must now be identity too
        pr, cid = jev_calib.apply("noul", [0.018, 0.982], mode="L")
        self.assertEqual(cid, "identity")
        self.assertEqual(pr, [0.018, 0.982])

    # ── 7. STRICT mode does not reject a task-tagged file ──────────────────
    def test_task_key_is_known_to_strict_loader(self):
        self.assertIn("task", jev_calib._KNOWN_KEYS)


    # ── 8. the SHIPPED default must be identity, not a stranger's fit ──────
    def test_shipped_calib_L_is_identity(self):
        """Regression guard for the defect this file exists to document.

        calib/calib-L.json used to ship a=0.6127, c=+0.6539 - a fit from BoolQ60/ARC60/SST5-60 that
        is applied to EVERY text task when STRATA_JEV_TASK is unset (legacy behaviour).  Applied to
        a 500-item banking77 set it moved the yes-rate from 67.6% to 80.4% and cost 9.6 accuracy
        points (p<0.001).  The default must be identity; the old fit lives in a task-tagged file the
        guard refuses.
        """
        import json as _json
        import pathlib as _pathlib
        here = _pathlib.Path(__file__).resolve().parent.parent / "calib"
        shipped = _json.loads((here / "calib-L.json").read_text())
        self.assertEqual(shipped["a_noul"], 1.0,
                         "the shipped default must not carry a cross-task slope")
        self.assertEqual(shipped["c_noul"], 0.0,
                         "the shipped default must not carry a cross-task intercept")
        # the historical fit survives, but only tagged - so task_mismatch() refuses it
        arch = _json.loads((here / "calib-L-boolq-arc-sst5.json").read_text())
        self.assertEqual(arch["task"], "boolq-arc-sst5")
        c = jev_calib.Calib(a_noul=arch["a_noul"], c_noul=arch["c_noul"], task=arch["task"])
        c.loaded = True
        self.assertNotEqual(jev_calib.task_mismatch(c, "banking77"), "",
                            "a tagged historical fit must be refused for another task")


if __name__ == "__main__":
    unittest.main()
