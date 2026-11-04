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

Strata already reuses the KV cache in two cases: a new prompt that *extends* the previous
one (the append case), and — importantly — a *mid-prompt edit*: upstream's strict mount
restores the deepest parked checkpoint that is a strict prefix of the new prompt, so an
edit at position E still mounts everything before the deepest surviving checkpoint. What
that mount cannot do is cross the edit: every checkpoint past E is invalid, and when edits
land near the tail (agents rewriting their working notes) the deepest prefix checkpoint is
shallow against the context length. What this PR adds over upstream is a partial-prefix
mount with a *denser anchor ladder* behind it. The state pool is only consulted when a
partial mount picks its restore anchor on a PARKED donor — the local strict search and the
parked strict pass never read it, and extending the ladder into the strict mount is future
work (§9):

- **Partial-prefix mount** (`--cache-suffix-reuse N`, opt-in): when a parked conversation's
  live image shares a longer prefix with the new prompt than any strict checkpoint does,
  and the session's state ladder has a restorable anchor deeper than both the strict
  match and the session's own resume floor, the engine mounts that image and restores the
  deepest checkpoint at or before the divergence — chosen only when its anchor beats
  max(strict match, flag-off floor), so it never restores shallower than flag-off.
  Only a parked LIVE image can partial-mount; a non-live partial would borrow and is
  skipped (fail-closed, r2 F5).
- **Dense tail** (`--prompt-cache-dense-tail N`): a position-dependent checkpoint cadence —
  sparse head, full cadence in the last N tokens — so edits that land near the end (most of
  them) have a close anchor on the *next* cold read.
- **State ladder** (`--prompt-cache-state-every N`): cheap state-only checkpoints at a fine
  cadence that deepen a partial mount's restore pick inside the gap the periodic anchors
  leave. The pool is read only by a partial mount on a parked donor; the strict mount —
  local or parked — never consults it (extending it there is future work, §9).

The feature family is opt-in, fails closed to the exact pre-feature re-read behavior, and
the default path is unchanged (pinned by `suffix_reuse_off_parity_test`).

Measured on the ledger probe (v0.1.45 stack): a cold 5.7k-token ledger prompt costs
**26.9 s**; a same-prompt replay mounts **98.4%** of the context and serves in **1.3 s**;
an *identical* re-send (no edit at all) is pure cache, **0.7 s**. These three are the
checkpoint *mount* working — upstream behaviour, not the ladder. A mid-prompt edit between
those extremes costs a re-read from the deepest checkpoint at or before the edit — with the
dense-tail + state-ladder configuration that is a bounded gap, not the whole prompt.

## 2. Background: what Strata caches, and the gap

Upstream ships two reuse mechanisms:

- **Conversation checkpoints** (`--prompt-cache N`, `--prompt-cache-every N`): the engine
  snapshots the KV at turn boundaries and chunk boundaries.
- **Conversation parking** (`--conversation-cache-mib N`): finished conversations' KV is kept
  in RAM so a later message in the same chat can mount it.

Both are **prefix-mounted**: a checkpoint is mountable only while it is a strict prefix of
the new prompt. A mid-file edit does not lose everything — the engine still mounts the
deepest checkpoint that survives as a prefix — but everything past the edit (including the
whole just-read tail the edit sits in) is invalid, and when the deepest surviving prefix
checkpoint is far from the edit the re-read is long.

## 3. Why this matters: agents that edit their context

The gap is not an edge case — it is the *dominant* pattern for agentic and context-management
workloads:

- **Agent loops with external memory.** An agent that keeps verbatim working notes re-sends
  them each turn; every rewrite invalidates every checkpoint inside the rewritten region.
- **Tool harnesses.** A file listing, a code window, or a scratchpad embedded in the prompt
  changes between calls while the surrounding system prompt and history stay identical.
- **Observed, not hypothesized.** In a 68-turn benchmark session against an engine with only
  prefix reuse, **41 of 66 requests (62%) got ~0% reuse** — each note rewrite left only a
  negligible prefix checkpoint to mount, effectively a full re-prefill. The same session's
  pure appends got 97–98%.

## 4. How it works

Given the old prompt O and the new prompt N with a longest common prefix of length P (the edit
at P):

1. **Strict prefix mount (continuation / append).** When a parked conversation or checkpoint is
   a strict prefix of N, the engine mounts it unchanged and reads only [L, n). This is the
   98.4% / 1.3 s case. The strict path is untouched by this PR's flag: no narrowing, no
   invalidation beyond upstream's own off-path erase (the strict splice hook was removed
   after review — `resume` always comes from a checkpoint that is a prefix of the new
   prompt, so it could never usefully narrow).
2. **Partial-prefix mount (mid-file edit).** When the strict mount fails, `best_with_partial`
   finds the candidate C with the longest LCP P (0 < P < L and P < n — a full-length LCP is
   the regenerate shape and is refused, restoring strict-mount behaviour exactly). The
   partial is chosen ONLY when its deepest restorable anchor at or before P — across the
   donor's periodic checkpoints and the session's state pool — is deeper than the strict
   match's s; a partial that would borrow (P below the donor's deepest checkpoint) is never
   eligible (the borrow path needs a checkpoint at exactly P). The running state at the old
   end is *stale* (it summarizes the old sequence) and is never reused: the engine restores
   the deepest SURVIVING checkpoint at or before P — across both checkpoint pools — and
   re-reads [ckpt, n) through the normal chunk loop (`narrowed_resume_target_state`).
   Fail-closed: when no candidate survives, the request degrades to a fresh prefill from 0;
   it never errors and never fabricates state.
3. **The gap is re-read, always.** Reading [P, n) on top of a checkpoint's state loses the gap
   [ckpt, P) from the GDN/PLE recurrence — attention sees the gap's rows through valid arena
   K/V, but the running state never ingested them, and count-style queries come back wrong
   (measured: an edited 161-row ledger answered "160"; gap 826 tokens). There is no replay
   shortcut: the read start equals the restored checkpoint's length
   (`read_from == restore_length`, enforced in the helper itself), byte-equal to the
   strict-path semantics.
4. **Checkpoint ladder (`--prompt-cache-state-every N`).** A side pool of state-only
   checkpoints (the recurrence state `checkpoint_save` already copies — gdn/ple/tails/dead/
   block_pos, no K/V, ~100–150 ms each) at a fine cadence. The restore pick takes the deepest
   anchor ≤ P across both pools, shrinking the fail-closed gap to the ladder's cadence below
   the edit — *while the pool can hold a checkpoint at that depth* (see §9: a state blob is
   ~20–25 MiB per 1K tokens in the code's own estimate, so at ~100K tokens a single
   checkpoint exceeds the pool's 2 GiB cap and the pool holds none — the gap then falls back
   to the periodic anchors). 0 = off = byte-identical. The pool is host-RAM capped at 2 GiB
   (oldest-first eviction).
5. **Dense tail (`--prompt-cache-dense-tail N`).** Checkpoints are taken sparsely through the
   head of the prompt (`max(every, 8192)`) and at the full cadence within the last N tokens —
   cheap cold read, dense anchors exactly where mid-text edits land. Sparse steps clamp to the
   zone start so the zone is never skipped.
6. **Resume wiring.** The chunk loop's read start resolves through `wire_splice_read_from`:
   the partial mount reads from the restored checkpoint's length (fail-closed, as above);
   every other path — flag off, append, strict mount — reads from `resume` exactly as
   upstream does (byte-identical, pinned by `suffix_reuse_off_parity_test`).

## 5. Measurements

- **Cold ledger prompt (5.7k tokens): 26.9 s.** Everything must be read once; the checkpoint
  cadence trades cold-read throughput for anchor density (§6).
- **Append / same-prefix turn: 1.3 s at 98.4% reuse.** The mount covers everything but the new
  text; only the appended tokens are read.
- **Identical re-send: 0.7 s.** Pure cache, nothing read.
- *(These three are the checkpoint mount working — upstream behaviour.)*
- **Mid-file edit: a re-read from the deepest checkpoint at/before the edit.** When a
  parked donor partial-mounts and the state ladder is at 256, the bound is one ladder
  step — *and only then when the state pool holds a checkpoint at that depth*; with no
  parked donor (editing the live conversation) the ladder is not read at all, and at
  very long contexts the pool may hold nothing (§9), so the bound falls back to the
  periodic-anchor spacing.
- **Ladder-specific numbers: withdrawn.** Earlier drafts credited the state ladder with a
  "2.1× mean reuse" gain and ContextBench family improvements (log-triage 21.5% → 77.2%,
  kv-store 43.1% → 72.3%). Those measurements predate the fix of a byte-cap comparison bug
  that left the state pool provably empty at run time, so whatever was measured it was not
  the ladder; the numbers are removed until re-measured on this code.
- **Off-parity:** at flag 0 the shared predicate returns false and the partial pass is never
  consulted — asserted by `suffix_reuse_off_parity_test` (byte-equal cache accounting, the
  chain untouched). Full flag-on/flag-off *output* parity on GPU is the orchestrator's
  verification step, not asserted by the CPU suite. The `[SFX-DBG]` debug prints are gated
  behind `STRATA_SFX_DBG`, so stderr is identical to upstream unless the env is set.

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

Refused at startup by the validator (the request never starts): `--kv-resident > 0`,
`--layer-split`, and any `--kv` format outside {fp16, int8, k8v4, q4_0} with the flag on;
`--prompt-cache-dense-tail > 0` without `--prompt-cache-every > 0`.

## 8. Verification

- CPU ctest: the six suffix suites — `suffix_reuse_test`, `suffix_reuse_off_parity_test`,
  `suffix_reuse_dense_tail_test`, `suffix_reuse_resume_narrowing_test`,
  `suffix_reuse_partial_prefix_test`, `suffix_reuse_continuation_and_anchor_test` — re-run
  after every commit.
- The three splice-only suites (`suffix_reuse_apply_test`, `suffix_reuse_int8_test`,
  `suffix_reuse_device_test`) tested the removed rotation layer and were deleted with it.

## 9. Limitations and future work

- **The state pool is host-side and its cap is its own.** It is budgeted at ~2 GiB of host
  RAM, is NOT counted by `--memory-limit-mib`, and is not parked with its conversation — a
  conversation that is evicted and returns finds the pool holding anchors for whatever ran
  in between. A state blob is ~20–25 MiB per 1K tokens (the code's own estimate), so at
  ~100K tokens a single checkpoint exceeds the cap and the pool holds none, while the cold
  read still pays every capture. `--prompt-cache-state-every` is opt-in (default 0 = off);
  ship it with that limitation stated.
- **The fail-closed gap [anchor, edit) is re-read through the normal chunk loop.** Shrinking it
  further requires advancing the GDN/PLE recurrence over parked arena K/V (a state-advance
  kernel — the FW-2c plan, parked; see `FW2C_DESIGN_NOTES.md`). Kill criterion there: if
  state-advance from parked K/V measures slower than the plain re-read at that depth, it is
  abandoned.
- **Partial mounts are conservative by construction.** A partial candidate is taken only
  when its deepest restorable anchor ≤ P beats the strict match's s, and a borrowing partial
  is never taken (r2 review findings 3/5). The practical consequence: same-conversation
  partials engage only through the state ladder — the flag alone (without
  `--prompt-cache-state-every`) mostly rides the strict mount.
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
