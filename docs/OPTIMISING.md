# Optimising accuracy on YOUR dataset

The decision surface is designed to work **out of the box, with no configuration and no calibration**.
This document is what to do *after* that, when you want better accuracy on your own data, in the order
that the measurements say pays best.

Nothing here is required to run the server. Everything here is a choice, and each step tells you the
measured size of the effect so you can decide whether it is worth your time.

## The default: generalised first

With no calibrator files and no task tag, every readout runs **identity**: probabilities are reported
as the model produced them, decisions are made where the model says so, and every answer carries
`"calib": "identity"` so you can see it did not invent a correction. Verified: an empty calibration
directory starts clean, reports `loaded: false` for every mode, and leaves both `noul` and `choice`
probabilities untouched.

That is the right default because it makes no claim about your data. A fit shipped by someone else is
a claim about *their* distribution, and the measurements below show how badly that goes.

## The levers, ranked by what they actually bought

| rank | lever | measured effect | cost |
|---|---|---|---|
| 1 | **derive a yes/no field from a choice question** | `is_card` .746 -> **.858** (p=1.2e-10), and one fewer readout | config only, **faster** |
| 2 | **calibrate on your own labelled items** | .779 -> **.836** on held-out (p=0.028) | 25-250 labelled items + a fit file |
| 3 | **define your categories in words** | neutral on a fixed 5-bucket task (.814 vs .822, p=0.712) but **moved 78/500 decisions** - it steers the boundary | a sentence per category |
| 4 | dynamic/self-tuning calibration | +6.4 on the yes/no readout (p=3e-5), unproven on choice | not shipped in this repository |

Everything after step 3 is refining a boundary. Steps 1 and 2 are what move accuracy.

## Step 0 - say what your categories mean

A `choice` question takes its option labels from `criteria`. Each entry may carry a one-line
description, which is how you tell the model what *you* mean by a category:

```json
"questions": {
  "group": {
    "type": "choice",
    "instructions": "Which group does this customer request belong to?",
    "criteria": {
      "card":      "the problem is with the card or with paying by card - chip, contactless, PIN, a declined payment",
      "top_up":    "money is being added to the account or a prepaid balance, including when paid for by card",
      "transfer":  "money is being sent to another person or account, or a transfer went wrong",
      "cash":      "cash in or out of the account - ATM withdrawals, cash deposits",
      "other":     "none of the above - refunds, fees, address or identity details, account access"
    }
  }
}
```

Omit the descriptions and the model chooses between bare labels. That is often fine - and on a
well-known taxonomy it changed nothing measurable - but when two of your categories are near
synonyms without a definition (ours were: `top_up` vs `transfer`, and an undefined `other`), the
definitions are the only thing that tells the model where the line is.

Be aware of what they do: in our test they made the model **more reluctant** to use one category
(37 answers moved `card` -> `other`), which traded top-1 accuracy for precision on a derived field.
Definitions steer the boundary, so measure the effect rather than assuming the direction.

## Step 1 - derive what you can instead of asking for it

> **Availability**: `derived_from` ships in the stacked derived-fields change, not in the base port.
> Check `grep derived_from serve/jev.py`; if it is absent in your build, skip to step 2 - the rest of
> this document stands on its own.

**This is the highest-value change available, and it is configuration, not training.**

If a yes/no field is really "is the answer category X", ask the choice question once and read the
yes/no field off it. The answer is more accurate *because the model can see the alternatives*, and it
costs no additional readout - roughly 35% less latency per item.

Measured on 500 labelled items: `is_card` direct **.746**, derived **.858** - 68 fixed, 12 broken,
p=1.2e-10, with one fewer readout per request.

```json
"is_card": {
  "type": "noul",
  "instructions": "Does this customer request concern a card?",
  "derived_from": {"qid": "group", "option": "card"}
}
```

Rules: the source must be a `choice` question in the same request, `option` must be one of its
labels, and chains are refused (a derived question cannot itself be a source). A bad reference is a
**422** naming the field, not a silent wrong answer.

Sanity check for whether this applies to you: our derived field's ceiling was exactly the parent's
accuracy - 70 of 71 remaining errors were group errors too. If your choice question is weak, this
inherits that weakness, which is why step 0 and step 1 are adjacent.

## Step 2 - build a labelled set

You cannot optimise what you cannot measure, and an answer key built by guessing is worse than none.

- **Size**: 25-100 balanced items is the practical floor for a calibration fit. Held-out accuracy by
  dev size measured 25 -> .824, 50 -> .824, 100 -> .822, 250 -> .809. More is better for measurement,
  not strictly needed for the fit.
- **Balance**: keep the natural class mix in your *evaluation* set. A class-balanced dev set is fine
  for fitting, with one caveat: the fitted offset encodes that balance, so evaluate on unmodified
  traffic or you will ship a prior that only matches your dev split.
- **Accept sets**: label the primary answer *and* the other answers you would accept. Our taxonomy
  needed this on 90 of 500 items - genuinely arguable cases - and reporting strict and lenient
  accuracy separately kept the argument honest instead of hiding it in one number.
- **Gold discipline**: write the labelling rule down before you label, and never let a substring or
  keyword heuristic produce gold. Our first key used `"card" in intent`, which scored correct answers
  as errors and understated group accuracy by ~8 points (.740 vs .824 strict). That single mistake cost
  more time than any modelling change in this project.

## Step 3 - measure a baseline before changing anything

Record the raw response for every item (probability, decision, confidence), keep the ids, and compute:

- accuracy, **and** Brier/ECE if you care about confidence being usable;
- paired comparisons only - same items, both arms, discordant pairs, McNemar. Comparing two accuracy
  numbers from two runs is not a comparison;
- strict and lenient accuracy if you have accept sets.

`scripts/eval_mf_set.py run` records an arm (per-item gold, decision, probability, confidence) and
`scripts/eval_mf_set.py compare` does the paired statistics for you: strict/lenient accuracy, Brier,
and the discordant-pair test. Build your gold yourself with the labelling rule written down - the
accept-set format is one row per item carrying `group`, `group_accept` (a list), `is_card` and
`is_card_accept`.

Beware two traps we fell into and paid for:

1. **Re-recording the same items is not an independent test.** Four "arms" over one 500-item set
   agreed on a threshold that vanished under a holdout split. Correlated errors look like replication.
2. **A threshold or parameter chosen by looking at the whole set is fitted, not measured.** Select on a
   dev half, evaluate once on the holdout. Ours picked 0.95 on dev and 0.30 on holdout - opposite ends
   of the grid, i.e. noise.

## Step 4 - ask whether calibration can help at all

Cheap, and it usually says no:

```
python3 scripts/calib_precondition.py noul   arm.json 25 0.524
python3 scripts/calib_precondition.py choice arm.json 50 0.524,0.118,0.122,0.072,0.164
#                                  <kind>  <arm>     <window> [declared target]
```

It compares your readout's operating statistic against the operating point you declare, and against
the oracle headroom a labelled correction could buy, and refuses when the gap sits inside its own
estimator noise (`headroom > 2 x se -> ADAPT`, otherwise `IDENTITY`).

If your readout is already at its ceiling, a calibrator cannot help and will hurt through its own
noise. That is not a hypothetical: it is why the shipped fits were removed rather than retuned.

## Step 5 - fit on your own data, with a holdout and a gate

```
python3 scripts/refit_calib.py fit --eval-json arm.json --task my-task
```

Stratified dev/holdout split, Platt fit, and **nothing is written unless the fit beats identity on both
accuracy and ECE on the holdout**. The numbers are printed either way, so a refusal is informative.

Useful properties of the fit, measured: the **intercept is the decision lever** - intercept-only ties
full Platt on accuracy while the slope only buys Brier/ECE. So a small number of labels gets you the
accuracy win.

## Step 6 - validate the file, then serve it tagged

```
python3 scripts/check_calib.py /path/to/calib-my-task -v
STRATA_JEV_TASK=my-task STRATA_JEV_CALIB_DIR=/path/to/calib-my-task ...
```

`check_calib.py` loads each file through the real loader and fails on keys the loader does not read, a
declared value that is not the value the loader ends up with, a degenerate temperature, or a `tuple`
letter that does not match the filename. Do not skip it: one of our files declared `"a"`/`"c"` when the
loader wanted `a_noul`/`c_noul`, so its fit was silently ignored for weeks while its temperature was
applied.

Set `STRATA_JEV_CALIB_STRICT=1` on any deployment that supplies a fit - it turns those silent drops
into refusals with a reason.

## What not to do, and why

- **Do not ship or reuse a fit fitted on another task.** Measured: a BoolQ/ARC/SST5 fit applied to a
  banking77 set cost **9.6 points** (.760 -> .664, p<0.001); a banking77 fit cost POPE **3.3** and GQA
  **8.3** points. The offset is a prior. Zeroing only `c` restored both.
- **Do not tune a threshold on the data you then report.** Selection bias, demonstrated above.
- **Do not expect calibration to fix missing evidence.** 61 of our 88 group errors were made
  *confidently* (top probability > 0.8). A boundary move cannot repair an error the model is sure about;
  that needs better questions (steps 0-1), more evidence in the prompt, or fine-tuning.
- **Do not move a question around hoping for a gain.** We tested port-style assistant-suffix placement
  against chat-turn placement: `is_card` .730 vs .758, 7 fixed / 21 broken, p=0.0125 - significantly
  worse.
- **Do not put a `score` readout on nominal categories.** `score` returns an expected class index,
  which is only meaningful on an ordinal scale; on nominal labels it is arithmetic on nothing.

## Reference: what the full set of levers looks like on one real task

500 labelled banking77 items, corrected gold, same binary:

| configuration | yes/no field | choice field |
|---|---|---|
| identity (default) | .746 | .824 strict / .872 lenient |
| shipped cross-task fit | .672 | .822 / .872 |
| derived from the choice field | **.858** | - |
| calibrated on this task (held out) | .836 | .834 (p=0.27, not significant) |
| categories defined in words | .874 | .814 / .862 |

The derived field is the only lever that both improved accuracy and made the system cheaper. That is
why it is step 1.
