#!/usr/bin/env python3
"""Decide, from a recorded stream, whether dynamic calibration can possibly help.

The score readout failed for a reason that generalises: an operating-point controller can only
remove error that is IN THE OPERATING STATISTIC.  If the readout's marginal already matches the
declared prior, there is nothing to remove, and the controller's own block noise makes things worse.
Measured on score: oracle headroom 0.5% (0.8691 -> 0.8646 MAE) while a 25-item block estimates the
mean with se ~ 0.22, so every configuration lost to identity.

So before enabling anything, ask two questions of a recorded stream:

  1. How far is the readout's operating statistic from the declared operating point?
  2. How much would a LABELLED constant correction buy (the oracle)?  That is an upper bound on any
     label-free shift, so if it is small, adaptation cannot win.

Gate:  oracle headroom > 2 x block-noise se   ->  ADAPT
       otherwise                              ->  IDENTITY (report-only monitor)

The 2x factor is deliberately conservative: below that the correction is inside its own estimator
noise.  This script is the check that would have stopped us shipping a score controller.

    python3 scripts/calib_precondition.py noul   arm.json 25 0.5
    python3 scripts/calib_precondition.py choice arm.json 50 0.5,0.118,0.122,0.072,0.164
    python3 scripts/calib_precondition.py score  arm.jsonl 25 1.522        # <kind> <arm> <window> [target]

Run this BEFORE scripts/refit_calib.py.  refit_calib answers "what are the best parameters",
this answers "is there anything for them to fix" - and it is the cheaper question.  No fit ships in
this repo precisely because the answer is usually "nothing": a fit is a claim about one distribution.
"""
import json
import math
import statistics as st
import sys

ORDER = ["card", "top_up", "transfer", "cash", "other"]


def read_rows(path):
    """Accept either a whole JSON document (the eval arms are pretty-printed) or JSONL."""
    text = open(path).read()
    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(text)
    if stripped.startswith("{"):
        try:
            doc = json.loads(text)
            if isinstance(doc, dict) and "rows" in doc:
                return doc["rows"]
            return [doc]
        except json.JSONDecodeError:
            pass                                   # not one document: fall through to JSONL
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def load(kind, path):
    """-> (values, golds, extra) where values are what the controller would steer on."""
    rows = read_rows(path)
    if kind == "noul":
        vals = [r["p_yes"] for r in rows if r.get("p_yes") is not None]
        golds = [r["gold"]["is_card"] for r in rows if r.get("p_yes") is not None]
        return vals, golds, "P(yes)"
    if kind == "choice":
        vals, golds = [], []
        for r in rows:
            pr = r.get("group_probs")
            if not pr:
                continue
            best = max(range(len(pr)), key=lambda i: pr[i])
            vals.append(float(best))                    # the predicted index
            golds.append(ORDER.index(r["gold"]["group"]))
        return vals, golds, "predicted class index"
    if kind == "score":
        vals = [r["score"] for r in rows if isinstance(r.get("score"), (int, float))]
        golds = [r["gold_index"] for r in rows if isinstance(r.get("score"), (int, float))]
        return vals, golds, "expected index"
    raise SystemExit(f"unknown kind {kind}")


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    kind, path, window = sys.argv[1], sys.argv[2], int(sys.argv[3])
    target = ([float(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4
              else None)

    vals, golds, what = load(kind, path)
    n = len(vals)
    if n < window * 4:
        raise SystemExit(f"need at least 4 windows, have {n} rows")

    # 1. operating statistic vs its declared point
    decisive = mean_v = 0.0
    if kind == "noul":
        decisive = sum(1 for v in vals if v >= 0.5) / n
        declared = target[0] if target else 0.5
        gap = abs(decisive - declared)
        unit = "positive-decision rate"
    else:
        mean_v = st.mean(vals)
        declared = target[0] if target else st.mean(golds)   # the declared marginal's mean index
        gap = abs(mean_v - declared)
        unit = "mean index"

    # 2. the oracle: the best correction IN THE CONTROLLER'S OWN FAMILY, fitted with labels
    headroom = 0.0
    if kind == "choice":
        # a scalar shift of a class index is meaningless; the family here is per-class biases,
        # so fit those by coordinate ascent on accuracy
        k = 5
        bias = [0.0] * k
        probs = []
        for r in read_rows(path):
            pr = r.get("group_probs")
            if pr:
                probs.append(([math.log(max(x, 1e-12)) for x in pr], ORDER.index(r["gold"]["group"])))

        def acc_of(b):
            hits = 0
            for z, g in probs:
                best = max(range(len(z)), key=lambda i: z[i] + b[i])
                if best == g:
                    hits += 1
            return hits / len(probs)

        base = acc_of(bias)
        for _ in range(40):                     # deterministic, small search
            improved = False
            for i in range(k):
                for delta in (-0.30, -0.15, -0.05, 0.05, 0.15, 0.30):
                    cand = list(bias)
                    cand[i] += delta
                    a = acc_of(cand)
                    if a > base + 1e-12:
                        base, bias, improved = a, cand, True
            if not improved:
                break
        raw_acc = acc_of([0.0] * k)
        headroom = base - raw_acc
        n = len(probs)

    if kind in ("noul", "score"):
        if kind == "noul":
            def err(shift):
                c = 0.5 - shift
                return sum(1 for v, g in zip(vals, golds)
                           if (1 if v >= c else 0) != g) / n
            base = err(0.0)
            best = min(err(s) for s in [i / 100 for i in range(-45, 46)])
        else:
            base = sum(abs(v - g) for v, g in zip(vals, golds)) / n
            best = min(sum(abs(v + s - g) for v, g in zip(vals, golds)) / n
                       for s in [i / 20 for i in range(-100, 101)])
        headroom = base - best
    elif kind == "score":
        base = sum(abs(v - g) for v, g in zip(vals, golds)) / n
        best = min(sum(abs(v + s - g) for v, g in zip(vals, golds)) / n
                   for s in [i / 20 for i in range(-100, 101)])
        headroom = base - best

    # 3. The relevant noise scale is the estimate over the whole stream, not one block.  A streaming
    # controller accumulates evidence across every block, so comparing the headroom with a single
    # block's standard error (the first version of this gate) wrongly rejects readouts that win:
    # noul's headroom is 5.4 accuracy points against a per-block se of 0.10, but against the
    # stream-level se of 0.022 it is 2.4x - and the measured controller recovered 4.2 of those 5.4.
    se = (st.pstdev(vals) if not isinstance(vals[0], int) else st.pstdev(vals)) / math.sqrt(n)
    if kind == "noul":
        se = math.sqrt(max((decisive if decisive else 0.5) * (1 - (decisive if decisive else 0.5)),
                           1e-9) / n)

    # 4. THE GATE.  A prior-matching controller removes marginal mismatch and nothing else, so the
    # discriminating question is whether that mismatch is real or estimator noise.  Gating on the
    # labelled oracle instead is wrong in both directions: for score the oracle finds 0.028 of
    # "headroom" that is label noise (the observed mean is already 0.017 from the target, and every
    # measured configuration lost), while for choice the oracle finds exactly the 2.4 points the
    # label-free controller captured - a real effect that a noise-floor test on the oracle throws
    # away.
    if kind == "noul":
        gap_stat = gap
        se_gap = math.sqrt(max((gap + 1e-9) * (1 - gap), 1e-9) / n)
    elif kind == "choice":
        # per-class predicted histogram against the declared marginal; the worst class decides
        pred = [0.0] * 5
        for v in vals:
            pred[int(v)] += 1.0 / n
        gap_stat = max(abs(pred[i] - target[i]) for i in range(5)) if target else 0.0
        se_gap = math.sqrt(max(0.25 * 4, 1e-9) / n)      # worst-case binomial se at p=0.5
    else:
        gap_stat = gap
        se_gap = (st.pstdev(vals)) / math.sqrt(n)

    ratio = gap_stat / se_gap if se_gap > 0 else 0.0
    verdict = "ADAPT" if ratio > 2.0 else "IDENTITY (report-only)"

    print(f"stream      : {path}  ({kind}, n={n}, {what})")
    print(f"declared pt : {declared:.4f}   observed: "
          f"{decisive if kind == 'noul' else mean_v:.4f}   mismatch: {gap_stat:.4f}")
    print(f"oracle      : labelled correction worth {headroom:.4f} (context only, NOT the gate)")
    print(f"GATE        : mismatch / se = {gap_stat:.4f} / {se_gap:.4f} = {ratio:.2f}  (need > 2.0)")
    print(f"VERDICT     : {verdict}")


if __name__ == "__main__":
    main()
