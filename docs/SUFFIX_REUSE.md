# Suffix-Reuse: Prefix Mounting for Edited Contexts

**Design and measurement document** for the `--cache-suffix-reuse` feature family on base v0.1.45:
the LCP prefix mount, the adaptive-cadence extension (`--prompt-cache-dense-tail`), and the
state-checkpoint ladder (`--prompt-cache-state-every`).
All numbers measured on: RTX 3090 (24 GB), Qwen3.8-Flash-Next abliterated GSQ-RCO IQ3_XXS
(2-file pack + PLE), 131k context, MTP spec-4, unless noted. Probes deterministic (temperature 0).

> **The K-rotation splice layer was removed after review (2026-11).** An expert review of the
> stacked `sfx` layers determined that the in-place rotation of cached K rows across a mid-prompt
> edit never saves work on any reachable path — the causal-attention K/V and the GDN/PLE
> recurrence disagree about the gap unless the recurrence is replayed, and replaying it costs the
> same re-read the rotation was meant to skip — and that it corrupted K in one reachable case.
> The splice headers, the apply call sites, the splice-only tests, and the never-shipped replay
> env gate are gone. What remains, and what this document describes, is the
> layer that actually runs and pays for itself: find how much of the old context still matches,
> mount the deepest valid checkpoint at or before the divergence, and re-read forward from there.

---

## 1. Summary

Strata already reuses the KV cache when a new prompt *extends* the previous one — the append
case. Without this feature it reuses nothing when the prompt is *edited in the middle*: one
changed token invalidates the entire mount and the whole prompt is re-read from scratch. The
engine now closes most of that gap with prefix mounting: when a prompt repeats an earlier one
with a change in the middle, it finds the longest common prefix (LCP), restores the deepest
checkpoint at or before the edit, and re-reads only the tail. The feature is opt-in
(`--cache-suffix-reuse N`), fails closed to the exact pre-feature re-read behavior, and the
default path is unchanged (pinned by `suffix_reuse_off_parity_test`).

Measured on the ledger probe (v0.1.45 stack): a cold 5.7k-token ledger prompt costs **26.9 s**;
a same-prompt replay mounts **98.4%** of the context and serves in **1.3 s**; an *identical*
re-send (no edit at all) is pure cache, **0.7 s**. A mid-prompt edit between those extremes
costs a re-read from the deepest checkpoint at or before the edit — with the dense-tail +
state-ladder configuration that is a bounded gap, not the whole prompt.

## 2. Background: what Strata caches, and the gap

Upstream ships two reuse mechanisms:

- **Conversation checkpoints** (`--prompt-cache N`, `--prompt-cache-every N`): the engine
  snapshots the KV at turn boundaries and chunk boundaries.
- **Conversation parking** (`--conversation-cache-mib N`): finished conversations' KV is kept
  in RAM so a later message in the same chat can mount it.

Both are **prefix-mounted**: the cached KV is valid only while the new prompt is the old prompt
plus new text at the end. A mid-file edit — rewrite a note, insert a row, change a line — makes
the cached sequence not-a-prefix, and the engine re-reads everything.

## 3. Why this matters: agents that edit their context

The gap is not an edge case — it is the *dominant* pattern for agentic and context-management
workloads:

- **Agent loops with external memory.** An agent that keeps verbatim working notes re-sends
  them each turn; every rewrite is a mid-prompt edit.
- **Tool harnesses.** A file listing, a code window, or a scratchpad embedded in the prompt
  changes between calls while the surrounding system prompt and history stay identical.
- **Observed, not hypothesized.** In a 68-turn benchmark session against an engine with only
  prefix reuse, **41 of 66 requests (62%) got 0% reuse** — each note rewrite triggered a full
  re-prefill. The same session's pure appends got 97–98%.

## 4. How it works

Given the old prompt O and the new prompt N with a longest common prefix of length P (the edit
at P):

1. **Strict prefix mount (continuation / append).** When a parked conversation or checkpoint is
   a strict prefix of N, the engine mounts it unchanged and reads only [L, n). This is the
   98.4% / 1.3 s case. A pure continuation must NOT be mis-identified as an edit: the plan's
   narrowing signal is LCP-based (`narrow_pos`), so an identical re-send mounts everything
   (pinned by `suffix_reuse_continuation_and_anchor_test` T9).
2. **Partial-prefix mount (mid-file edit).** When the strict mount fails, `best_with_partial`
   finds the checkpoint C with the longest LCP P (0 < P < L). The K/V cells for [0, P) are
   valid by the LCP; the running state at the old end is *stale* (it summarizes the old
   sequence) and is never reused. The engine restores the deepest SURVIVING checkpoint at or
   before P — across both checkpoint pools — and re-reads [ckpt, n) through the normal chunk
   loop (`narrowed_resume_target_state`). Fail-closed: when no candidate survives, the request
   degrades to a fresh prefill from 0; it never errors and never fabricates state.
3. **The gap is re-read, always.** Reading [P, n) on top of a checkpoint's state loses the gap
   [ckpt, P) from the GDN/PLE recurrence — attention sees the gap's rows through valid arena
   K/V, but the running state never ingested them, and count-style queries come back wrong
   (measured: an edited 161-row ledger answered "160"; gap 826 tokens). There is no replay
   shortcut: the partial mount's read start equals the restored checkpoint's length
   (`read_from == restore_length`), byte-equal to the strict-path semantics.
4. **Checkpoint ladder (`--prompt-cache-state-every N`).** A side pool of state-only
   checkpoints (the recurrence state `checkpoint_save` already copies — gdn/ple/tails/dead/
   block_pos, no K/V, ~100–150 ms each) at a fine cadence. The restore pick takes the deepest
   anchor ≤ P across both pools, shrinking the fail-closed gap to ≤ N tokens below the edit.
   0 = off = byte-identical. The pool is host-RAM capped at 2 GiB (oldest-first eviction).
5. **Dense tail (`--prompt-cache-dense-tail N`).** Checkpoints are taken sparsely through the
   head of the prompt (`max(every, 8192)`) and at the full cadence within the last N tokens —
   cheap cold read, dense anchors exactly where mid-text edits land. Sparse steps clamp to the
   zone start so the zone is never skipped.
6. **Narrowed resume.** The hook narrows `resume` to the edit position and filters the
   checkpoint chain to anchors at or below it; the chunk loop then reads from the narrowed
   target via `wire_splice_read_from`. Off (the default), behavior is byte-identical to today
   (pinned by `suffix_reuse_off_parity_test`).

## 5. Measurements

- **Cold ledger prompt (5.7k tokens): 26.9 s.** Everything must be read once; the checkpoint
  cadence trades cold-read throughput for anchor density (§6).
- **Append / same-prefix turn: 1.3 s at 98.4% reuse.** The mount covers everything but the new
  text; only the appended tokens are read.
- **Identical re-send: 0.7 s.** Pure cache, nothing read.
- **Mid-file edit: a re-read from the deepest checkpoint at/before the edit.** With the
  state ladder at 256, that bound is 256 tokens, not `every`-spacings. ContextBench campaign
  (4 families, seed 42, dense-tail + state-256): reuse rose exactly where predicted
  (log-triage 21.5% → 77.2%, kv-store 43.1% → 72.3%) with every family at or above its base
  score; multi-seed (42/43/44) shows the state ladder roughly doubling mean reuse (2.1×)
  while reward-neutral.
- **Off-parity:** flag off vs flag on with no edits must produce identical output and identical
  cache accounting — asserted, not assumed (`suffix_reuse_off_parity_test`).

## 6. The honest cost: checkpoint granularity

Reuse requires a checkpoint at or before the edit position. With `--prefill auto` (one big
chunk) there are no mid-prompt checkpoints and mid-edit reuse does not fire (appends still
reuse). The proven configurations are `--prefill 1024 --prompt-cache-every 1024` (uniform
density, expensive cold read) or `--prefill auto --prompt-cache-dense-tail 4096
--prompt-cache-state-every 256` (sparse head + dense tail + state ladder). These trade-offs are
why the feature is opt-in and the knobs are exposed.

## 7. Configuration

```
--cache-suffix-reuse 64          # the feature (opt-in; 0 = off, default)
--prompt-cache-state-every 256   # state-checkpoint ladder (0 = off = prior behavior)
--prompt-cache-dense-tail 4096   # adaptive cadence (0 = off = uniform cadence)
--conversation-cache-mib 4096    # upstream flag; parking must be on for cross-turn mounts
--prompt-cache 6                 # upstream default (>= 2 required by the validator)
--prefill 1024 --prompt-cache-every 1024   # uniform density (or dense-tail instead)
```

Refused (logged, falls back to full re-read, never crashes): `--kv-resident > 0`,
`--layer-split` with the flag on.

## 8. Verification

- CPU ctest: the six suffix suites — `suffix_reuse_test`, `suffix_reuse_off_parity_test`,
  `suffix_reuse_dense_tail_test`, `suffix_reuse_resume_narrowing_test`,
  `suffix_reuse_partial_prefix_test`, `suffix_reuse_continuation_and_anchor_test` — re-run
  after every commit.
- The three splice-only suites (`suffix_reuse_apply_test`, `suffix_reuse_int8_test`,
  `suffix_reuse_device_test`) tested the removed rotation layer and were deleted with it.
- Off-parity (reuse off vs on, identical output) is asserted by test, not inspection.

## 9. Limitations and future work

- The fail-closed gap [anchor, edit) is re-read through the normal chunk loop. Shrinking it
  further requires advancing the GDN/PLE recurrence over parked arena K/V (a state-advance
  kernel — the FW-2c plan, parked; see `FW2C_DESIGN_NOTES.md`). Kill criterion there: if
  state-advance from parked K/V measures slower than the plain re-read at that depth, it is
  abandoned.
- Streaming-ring (`--kv-resident`) and layer-split layouts remain refused.
- HIP compile not tested (no AMD hardware; the remaining code is host-side C++).

## References

1. CacheSlide: Unlocking Cross Position-Aware KV Cache Reuse for Accelerating LLM Serving —
   Liu et al., USENIX FAST '26. https://www.usenix.org/system/files/fast26-liu-yang.pdf
2. llama.cpp issue #18497 — cache-reuse effectiveness on quantized caches.
   https://github.com/ggml-org/llama.cpp/issues/18497
3. llama.cpp discussion #8860 — prefix cache reuse semantics.
   https://github.com/ggml-org/llama.cpp/discussions/8860
4. Adaptive KV Cache Reuse for Fast Long-Context LLM Serving — arXiv:2605.24022.
   https://arxiv.org/abs/2605.24022
5. vLLM prompt-cache reuse RFC — absolute-position mismatch discussion.
   https://github.com/vllm-project/vllm/issues/2614
