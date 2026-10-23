#!/usr/bin/env python3
"""maxfield_probe - multi-field latency to the field cap, one state, N fields per request.

Why it exists: the five-field arm table cannot separate "per request" from "per field" — the difference is
inside the run-to-run spread.  At 8..32 fields the per-field cost dominates, so the arms separate and the
shape of the curve says which one wins at the field counts a real caller uses.

The fields are generated, not curated: field i is a distinct question on the same state, cycling noul /
choice / score so the readout paths are mixed the way a real request is.  Everything is measured on one
request per repetition, with the first repetition discarded as warmup, and the state is rebuilt for each
repetition so a cache hit cannot be mistaken for a win.

  maxfield_probe.py --base URL --key K --out FILE --counts 4,8,12,16,24,32 --reps 3
"""
import argparse
import json
import statistics
import time
import urllib.request

STATE = {"engine": "hot", "load": 0.9, "queue": 12, "tier": "gold",
         "notes": "the intake queue is backed up and the engine has been at 90% for twenty minutes"}


def make_fields(n):
    """N distinct fields on one state: every third one is noul, choice or score."""
    out = {}
    for i in range(n):
        qid = "f%02d" % i
        if i % 3 == 0:
            out[qid] = {"type": "noul", "instructions": "Is condition %d true?" % i}
        elif i % 3 == 1:
            out[qid] = {"type": "choice", "instructions": "Which option applies to item %d?" % i,
                        "criteria": {"alpha": 1, "beta": 2, "gamma": 3}}
        else:
            out[qid] = {"type": "score", "instructions": "Rate item %d." % i,
                        "criteria": ["low", "mid", "high"]}
    return out


def post(base, key, body, timeout=1800):
    req = urllib.request.Request(base + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return (time.time() - t0) * 1000.0, r.status, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--counts", default="4,8,12,16,24,32")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    out = {"label": args.label, "base": args.base, "cases": []}
    for n in [int(x) for x in args.counts.split(",") if x.strip()]:
        fields = make_fields(n)
        runs = []
        for i in range(args.reps + 1):                     # first one is warmup
            try:
                ms, code, body = post(args.base, args.key, {"state": STATE, "questions": fields})
            except Exception as e:                         # noqa: BLE001 - reported, never hidden
                print("[%s] fields=%d rep=%d EXC %s" % (args.label, n, i, e), flush=True)
                runs.append({"code": 0, "ms": None, "error": str(e)})
                continue
            usage = body.get("usage", {}) if isinstance(body, dict) else {}
            print("[%s] fields=%d rep=%d code=%d ms=%.1f tokens=%s" %
                  (args.label, n, i, code, ms, usage.get("prompt_tokens")), flush=True)
            if i == 0:
                continue
            runs.append({"code": code, "ms": round(ms, 1), "usage": usage})
        ok = [r["ms"] for r in runs if r.get("ms") is not None]
        case = {"fields": n, "runs": runs,
                "ms_median": round(statistics.median(ok), 1) if ok else None,
                "usage": runs[-1].get("usage") if runs and runs[-1].get("usage") else None}
        out["cases"].append(case)
        print("[%s] fields=%d median=%s tokens=%s" %
              (args.label, n, case["ms_median"], (case["usage"] or {}).get("prompt_tokens")), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
