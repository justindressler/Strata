# jev-port calibration: NO FITS SHIP, here are the tools to fit your own

**This directory contains no calibrator files, by design.** The server runs **identity** until you fit
a calibrator on data from your own task. That is not a missing feature, it is the measured best
default.

Why: a fit is a claim about a data distribution. `a_noul`/`c_noul` are a Platt rescale whose offset is
a **decision threshold** - it encodes "how often is the answer yes in *my* data". Applied to a
different task it is not neutral, it is wrong in a specific direction, and it is confidently wrong.

Measured, all on the same binary, only the calibrator differing:

- **A cross-task fit cost 9.6 accuracy points.** `calib-L.json` was fitted on BoolQ60/ARC60/SST5-60 and
  applied to every text task. On 500 labelled banking77 items it moved the yes-rate from 67.6% to
  80.4% and accuracy 0.760 -> 0.664 (p<0.001). **Doing nothing beat what shipped.**
- **A banking77 fit cost other tasks 3.3 and 8.3 points** (POPE, GQA). Zeroing only `c` restored both.
  The slope is roughly portable; **the offset never is**.
- **The tasks the shipped fits came from no longer need them.** Re-run on the current binary: BoolQ
  .910 -> .910, ARC 1.000 -> 1.000, SST-5 .520/.530 with the refit *hurting* test ECE (.127 -> .140).
  The readout improved underneath the calibrator; headroom there is ~0 and refits come back near
  identity (a=1.009 c=.0037).

So the repository ships the **procedure** and not the **parameters**. Anything you fit is yours, for
your task, evaluated on items it never saw.

This file covers calibration only. For the end-to-end process - defining your categories, deriving
fields instead of asking for them, building a labelled set, and measuring - read
[**docs/OPTIMISING.md**](../docs/OPTIMISING.md). Calibration is step 5 of 6, and step 1 buys more than it does.

## The procedure, in the order that saves the most time

**Step 0 - ask whether calibration can help at all.** This is the cheap question and it usually
answers "no".

```
python3 scripts/calib_precondition.py noul   arm.json 25 0.524
python3 scripts/calib_precondition.py choice arm.json 50 0.524,0.118,0.122,0.072,0.164
#                                  <kind>  <arm>     <window> [declared target]
```

It compares the readout's operating statistic against the declared operating point and against the
oracle headroom, and refuses to recommend a fit when the gap is inside its own estimator noise
(`headroom > 2 x se -> ADAPT`, otherwise `IDENTITY`). Measured verdicts on real arms: text noul
10.33 -> ADAPT, text choice 2.73 -> ADAPT, score 0.35 -> IDENTITY. The score readout has 0.5% oracle
headroom while a 25-item block estimates its mean with se ~0.22, so every adaptive configuration
tested lost to identity.

**Step 1 - collect labelled items.** 25-100 balanced items is the practical floor; held-out accuracy
by dev size measured 25 -> .824, 50 -> .824, 100 -> .822, 250 -> .809.

**Step 2 - fit with a real dev/holdout split, and require the gate to pass.**

```
python3 scripts/refit_calib.py fit --eval-json arm.json --task my-task
```

It splits stratified dev/holdout, fits Platt, and writes nothing unless the fit beats identity on
**both** accuracy and ECE on the holdout. It prints the numbers either way. A fit that cannot clear
the gate is not worth its maintenance cost.

**Step 3 - serve it, tagged.**

```
STRATA_JEV_TASK=my-task STRATA_JEV_CALIB_DIR=/path/to/calib-my-task ...
```

The tag is the contract: with `STRATA_JEV_TASK` set, a fit whose recorded `task` differs is refused
and the response says `"calib": "identity"`. With no tag set the loader keeps legacy behaviour and
applies whatever loads - which is exactly why nothing ships here.

**Step 4 - validate the file against the loader that will read it.**

```
python3 scripts/check_calib.py /path/to/calib-my-task -v
```

By eye you cannot catch a fit that is silently ignored. `check_calib.py` loads each file through
`serve/jev_calib.py` and fails on: keys the loader does not read, a declared value that is not the
value the loader ends up with, a degenerate `T` (<=0.05 or >=10), or a `tuple` letter that does not
match the filename. That first check exists because `calib-S.json` once shipped `"a"`/`"c"` - keys the
loader never reads - so its noul fit was silently identity while its `T_choice=0.05` was honoured
(exponent 20, near one-hot).

## The other tooling in this directory

- `bench_suite.py` - the live calibration suite (text/image/real). Fits Platt (a,c) for noul and
  temperature for choice/score on dev splits, evaluates on held-out test with top-label ECE15 and mean
  NLL, writes per-item jsonl logs. Guards: refuses Platt slope a<=0 (labels/prompt inverted) and flags
  T at its search boundary.
- `convert_benchmarks.py` - POPE adversarial / CV-Bench / CountBenchQA -> item JSONs (banned-field
  selftest, sha256 image manifest, dev/test image-disjoint, POPE sampled by image at exact 50/50).
- `blockers_gpu.py` - phase-C fits over the real suites.
- `token_audit.py` - readout token accounting.
- `scripts/refit_calib.py`, `scripts/check_calib.py`, `scripts/calib_precondition.py`,
  `scripts/fit_calib_mf.py`, `scripts/transfer_check.py` - the fit/gate/validate/transfer tools.

## What a calibrator file looks like

```json
{
  "tuple": "qfn-iq3_xxs|jev-port|none|v1|L|text",
  "task": "my-task",
  "a_noul": 1.0,
  "c_noul": 0.0,
  "T_choice": 1.0,
  "T_score": 1.0,
  "provenance": "fit <date> on <dev set>, holdout acc .xxx -> .yyy, gate PASS"
}
```

Only `a_noul`/`c_noul`/`T_choice`/`T_score` are read; `tuple` binds the file to a readout letter and
`task` binds it to a task tag. Identity is `a_noul=1.0, c_noul=0.0, T_choice=1.0, T_score=1.0`.

## The fits this directory used to contain

Kept as a worked example of the failure mode, not as usable parameters. Do not copy these numbers;
they are wrong for your data.

| file | a_noul | c_noul | T_choice | T_score | provenance |
|---|---|---|---|---|---|
| calib-L | 0.6127 | 0.6539 | 1.0557 | 2.263 | BoolQ60/ARC60/SST5-60, 2026-10-15 |
| calib-D | 1.0571 | -0.1720 | 1.5456 | 0.6555 | POPE-adv/CV-Bench/CountBenchQA |
| calib-S | 0.4892 | 0.8313 | 0.05 | 1.8702 | blockers_gpu phase C - never applied, see below |

Two lessons are embedded in that table:

- `calib-S` shipped `"a"`/`"c"`, which the loader does not read. Its fit was **never applied**; only
  its temperatures were. A shipped parameter set can be inert and nobody notices for weeks.
- The image fits (`calib-D`) were re-measured and change no decision while improving Brier by
  .0046/.0315/.0209 and ECE by -.0075/+.0100/-.0583 - i.e. **at their ceiling**. Even a good fit buys
  almost nothing once the readout is good.

## Rules the tooling enforces or warns about

- Fit per (tuple, task, mode) and evaluate on items the fit never saw. If the dev split is saturated
  (acc 1.000, NLL ~.00x) then `T` is unidentifiable, and a fitter reporting 0.05 or 5.03 there is
  reporting its own search bound, not a temperature.
- Never fit on calibrated output. Move any `calib-*.json` aside to serve identity for a raw->fit
  measurement, or invert the known transform exactly first.
- Never carry `c` across tasks. The slope is even approximately portable; **the offset is a prior**.
- Report held-out numbers only. Every figure measured on its own dev split was too optimistic, and the
  ones re-run held-out did not replicate.
- `STRATA_JEV_CALIB_STRICT=1` refuses a file whose keys the loader would silently drop (the `a`/`c`
  vs `a_noul`/`c_noul` trap) and refuses a degenerate `T`. Recommended for any deployment that does
  supply a fit.
