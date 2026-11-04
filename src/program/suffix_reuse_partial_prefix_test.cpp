// src/program/suffix_reuse_partial_prefix_test.cpp - phase 3b: T12-T15.  The partial-prefix
// mount's pure helpers and the flag-gated `best_with_partial` path.  Header-only / CPU - no
// CUDA, no <cuda_runtime.h>, no QsaState.  Pins the exact behaviour the runtime integration
// site in generate.cpp relies on.
//
// CONTEXT.  Phase 3a shipped the strict-mount splice (T9 continuation parity + T10 anchor
// engagement).  On ContextBench traffic the strict mount never fires for the 44/71 mid-file-
// edit requests because the previous conversation's END is no longer a strict prefix of the
// new prompt.  Phase 3b adds a flag-gated partial-prefix mount: when the strict mount fails,
// find the checkpoint C with the longest LCP P (0 < P < L) and mount C.  The K/V cells for
// [0, P) are valid from the restore, the chunk loop reads [P, n), and the existing splice
// machinery handles the K rotation for the unchanged tail [P, L) and the prefill of [P, n).
//
// OFF-FLAG BYTE-IDENTITY.  `ConversationCache::best_with_partial(0, ...)` MUST return the
// same Match as `best()` (the same loop, same `conversation_prefix` call, same tie-break).
// The new `partial` and `checkpoint_length` fields default to false / 0 on the strict path,
// so downstream consumers that don't read them see no change.  T13 pins this directly.
//
// T12 PARTIAL-PREFIX MATCH.  Checkpoint {A B C D} (length 4), prompt {A B X Y} (length 4).
// Strict match: 0.  Partial match: LCP = 2 (matches A, B; diverges at index 2).  Asserts:
//   - best_with_partial(flag>0, prompt, ...) returns Match{index, tokens=2, partial=true,
//     checkpoint_length=4, live=appropriate}
//   - suffix_reuse_partial_apply(C.tokens, prompt, L=4, P=2, chain) returns:
//       fired = true
//       plan_edit_pos = 2 (= the LCP)
//       restore_anchor_length = 4 (= the full checkpoint's length)
//       old_suffix_len = 0 (C's tail [C D] vs prompt's tail [X Y] - no common suffix)
//       chain drops any entry with length > 2
//
// T12 variant: partial match WITH a common tail (the K rotation engages - pure insert of 1
// token at position 2: OLD = {1 2 3 4 5} (5), NEW = {1 2 90 3 4 5} (6), LCP=2, suffix=3,
// uniform=true, delta=+1).
//
// T13 OFF-FLAG PARITY.  Same inputs as T12 with flag = 0:
//   - best_with_partial(0, ...) returns the same as best() - strict search, 0 tokens
//   - the partial helper is NOT called (the runtime only calls it when the cache returned
//     a partial match - at flag=0 the cache never returns a partial match)
//
// T14 BEST-OF-N.  Multiple checkpoints with different LCPs - the longest LCP wins.  Strict
// match still preferred over partial (strict > partial always, even if strict is shorter).
//
// T15 IMAGES / GEOMETRY MISMATCH.  Partial match is REFUSED when images mismatch at positions
// below the LCP: a checkpoint with an image at position 1 cannot mount a prompt that doesn't
// have that image (the LCP is 2 but the image is at position 1 which is below the LCP).  The
// fail-closed case: no mount, no error.

#include "strata/core/conversation_cache.hpp"
#include "strata/program/suffix_reuse.hpp"

#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

using namespace strata::core;
using strata::program::suffix_reuse::CheckpointLength;
using strata::program::suffix_reuse::SuffixHookOutcome;
using strata::program::suffix_reuse::suffix_reuse_partial_apply;
using strata::program::suffix_reuse::compute_suffix_plan;

namespace {
int g_fail = 0;
int g_pass = 0;
void check(bool ok, const char* what) {
    std::printf("  %-90s %s\n", what, ok ? "ok" : "FAIL");
    if (ok) ++g_pass; else ++g_fail;
}
SavedConversation image(std::initializer_list<int32_t> ids, bool cvec = true) {
    SavedConversation s;
    s.live.ids = ids;
    s.live.gdn.resize(64, 7);
    s.cvec = cvec;
    return s;
}
ConversationCheckpoint ckpt(std::initializer_list<int32_t> ids) {
    ConversationCheckpoint c;
    c.ids = ids;
    c.gdn.resize(64, 1);
    return c;
}
}  // namespace

int main() {
    std::printf("suffix_reuse_partial_prefix_test (T12 partial-prefix mount, T13 off-flag parity, T14 best-of-N, T15 image mismatch)\n");
    // ============================================================
    // T12: PARTIAL-PREFIX MATCH
    // ============================================================
    // Checkpoint {A B C D} (length 4).  Prompt {A B X Y} (length 4).
    // Strict match: 0 (prompt[2]=X != C, not a prefix).
    // Partial match: LCP = 2 (A, B match; index 2 diverges: C vs X).
    // Common tail from back: [C, D] vs [X, Y] -> 0 (no match).
    // Splice plan: narrow_pos = 2, old_suffix_len = 0, uniform = false (no deltas to compute).
    {
        auto s = image({1, 2, 3, 4});
        s.checkpoints = {ckpt({1, 2, 3, 4})};
        ConversationCache cache(1 << 20, 4);
        cache.put(std::move(s));

        const std::vector<int32_t> prompt = {1, 2, 90, 91};   // [A B X Y]
        // best() (strict): no match (the live and checkpoint are full of {1,2,3,4}, prompt diverges at 2).
        const auto strict = cache.best(prompt, {}, true);
        check(!strict.partial && strict.tokens == 0,
              "T12 setup: strict best() returns no match for partial-prefix shape");

        // best_with_partial(flag=64, ...): partial match, LCP=2, ckpt length=4.
        // The session's state pool (r2 F3) supplies the restorable anchor at 2 <= P -
        // without an anchor deeper than the strict s (0 here) the partial is not chosen.
        const std::vector<int64_t> state_pool = {2};
        const auto m = cache.best_with_partial(64, prompt, {}, true, state_pool);
        check(m.partial,
              "T12.1: best_with_partial(flag>0) returns partial=true for {A B C D} / {A B X Y}");
        check(m.tokens == 2,
              "T12.2: match.tokens == 2 (the LCP)");
        check(m.checkpoint_length == 4,
              "T12.3: match.checkpoint_length == 4 (the full checkpoint's length)");
        check(m.tokens < m.checkpoint_length,
              "T12.4: LCP < checkpoint_length (the partial-mount invariant)");
        // r2 F3: with NO state anchor deeper than s the partial is refused - the donor's
        // periodic checkpoints at or before P are prefixes the strict pass already sees.
        const auto m_nostate = cache.best_with_partial(64, prompt, {}, true);
        check(!m_nostate.partial && m_nostate.tokens == 0,
              "T12.4b: no state anchor -> strict stands (s=0); partial never chosen (r2 F3)");

        // The partial helper: fire, plan, anchor.
        const std::vector<int32_t> ck_tokens = {1, 2, 3, 4};
        std::vector<CheckpointLength> chain = { {4, 0} };
        const SuffixHookOutcome outcome = suffix_reuse_partial_apply(
            ck_tokens, prompt, /*L=*/4, /*P=*/2, chain);
        check(outcome.fired,
              "T12.5: partial_apply fired (valid partial shape)");
        check(outcome.plan_edit_pos == 2,
              "T12.6: outcome.plan_edit_pos == 2 (the LCP)");
        check(outcome.restore_anchor_length == 4,
              "T12.7: outcome.restore_anchor_length == 4 (the partial-mount anchor)");
        // The tail fields live on the PLAN (r2 F7 deleted the outcome's dead copies).
        const auto plan = compute_suffix_plan(ck_tokens, prompt);
        check(plan.old_suffix_len == 0,
              "T12.8: plan.old_suffix_len == 0 ([C D] vs [X Y] - no common tail)");
        check(plan.new_suffix_len == 0,
              "T12.9: plan.new_suffix_len == 0 (no common tail)");
        check(!plan.uniform,
              "T12.10: plan.uniform == false (no deltas to compute)");
        check(outcome.chain.size() == 1 && outcome.chain[0].length == 4,
              "T12.11: chain keeps the anchor (length 4 == L); non-anchor entries with length > P are dropped");

        // The runtime's narrowing predicate at generate.cpp:9074: `if (outcome.plan_edit_pos < resume)`.
        // For the partial case, resume is set to P (= 2) by the runtime before the splice hook runs;
        // plan_edit_pos == 2 == resume, so the predicate is false.  The runtime uses
        // `restore_anchor_length` to set resume = L (= 4) AFTER the splice hook returns.
        // (The runtime's flow: cache.best_with_partial -> set resume=P -> call splice -> read restore_anchor_length -> set resume=L -> restore C -> chunk loop reads P.)
        // We pin the helper's contract: plan_edit_pos == P, restore_anchor_length == L.
        check(outcome.plan_edit_pos != outcome.restore_anchor_length,
              "T12.12: partial case - plan_edit_pos (P) != restore_anchor_length (L) [sentinel]");

        // The (removed) K-rotation layer was historically gated on the plan's
        // old_suffix_len; the layer and its outcome fields are gone (r2 F7) - the
        // prefill recomputes K+V for [read_from, n) anyway.
    }

    // ============================================================
    // T12 variant: partial match WITH a common tail (a pure insert of 1; the tail fields
    // are pinned on the PLAN - the outcome no longer carries them, r2 F7)
    // ============================================================
    // OLD = {A B C D E} (length 5), NEW = {A B X C D E} (length 6 - pure insert at position 2).
    // LCP = 2 (A, B match; index 2 diverges: C vs X).
    // Common tail from back: [C, D, E] vs [C, D, E] -> 3.
    {
        const std::vector<int32_t> ck_tokens = {1, 2, 3, 4, 5};     // OLD: [A B C D E]
        const std::vector<int32_t> prompt    = {1, 2, 90, 3, 4, 5}; // NEW: [A B X C D E]
        std::vector<CheckpointLength> chain;
        const SuffixHookOutcome outcome = suffix_reuse_partial_apply(
            ck_tokens, prompt, /*L=*/5, /*P=*/2, chain);
        check(outcome.fired, "T12v.1: partial_apply fired (LCP=2, suffix=3)");
        check(outcome.plan_edit_pos == 2, "T12v.2: plan_edit_pos == 2 (LCP)");
        check(outcome.restore_anchor_length == 5, "T12v.3: restore_anchor_length == 5 (anchor)");
        const auto plan = compute_suffix_plan(ck_tokens, prompt);
        check(plan.old_suffix_len == 3,
              "T12v.4: plan.old_suffix_len == 3 ([C D E] common tail in OLD)");
        check(plan.new_suffix_len == 3,
              "T12v.5: plan.new_suffix_len == 3 ([C D E] common tail in NEW)");
        check(plan.uniform,
              "T12v.6: plan.uniform == true (pure insert of 1, all deltas == +1)");
        check(plan.delta == 1,
              "T12v.7: plan.delta == +1 (insert of X shifts C,D,E from OLD [2,4) to NEW [3,5))");
        check(plan.deltas.empty(),
              "T12v.8: uniform -> deltas table empty (single rotation suffices)");
    }

    // ============================================================
    // T12 variant: partial match WITH a common tail of a different size (non-uniform)
    // ============================================================
    // OLD = {A B C D E F} (6), NEW = {A B X Y C D E F} (8 - insert of 2 at position 2).
    // LCP = 2, suffix = 4 ([C D E F]).
    // uniform_delta = n_new - n_old = 2.  For each suffix token: new_pos = 8-4+i, old_pos = 6-4+i.
    // deltas = [2, 2, 2, 2].  uniform=true, delta=+2.
    {
        const std::vector<int32_t> ck_tokens = {1, 2, 3, 4, 5, 6};
        const std::vector<int32_t> prompt    = {1, 2, 90, 91, 3, 4, 5, 6};
        std::vector<CheckpointLength> chain;
        const SuffixHookOutcome outcome = suffix_reuse_partial_apply(
            ck_tokens, prompt, /*L=*/6, /*P=*/2, chain);
        const auto plan = compute_suffix_plan(ck_tokens, prompt);
        check(outcome.fired && plan.uniform && plan.delta == 2,
              "T12v2.1: pure insert of 2 -> delta=+2 (on the plan; r2 F7)");
    }

    // ============================================================
    // T13: OFF-FLAG BYTE-IDENTITY
    // ============================================================
    // Same shape as T12, flag=0: best_with_partial returns the same as best() (no partial).
    {
        auto s = image({1, 2, 3, 4});
        s.checkpoints = {ckpt({1, 2, 3, 4})};
        ConversationCache cache(1 << 20, 4);
        cache.put(std::move(s));

        const std::vector<int32_t> prompt = {1, 2, 90, 91};   // [A B X Y]
        // flag=0: byte-identical to best() - no partial match, no partial fields touched.
        const auto m0 = cache.best_with_partial(0, prompt, {}, true);
        const auto mb = cache.best(prompt, {}, true);
        check(!m0.partial,
              "T13.1: flag=0 -> partial=false (byte-identical to best()'s default)");
        check(m0.tokens == mb.tokens && m0.index == mb.index && m0.live == mb.live,
              "T13.2: flag=0 -> match fields == best() (index, tokens, live)");
        check(m0.tokens == 0,
              "T13.3: flag=0 + no strict match -> tokens == 0 (the runtime sees no mount)");

        // flag<0 (canonical "flag absent"): same as flag=0.
        const auto m_neg = cache.best_with_partial(-1, prompt, {}, true);
        check(!m_neg.partial && m_neg.tokens == 0,
              "T13.4: flag<0 (canonical 'flag absent') -> byte-identical to flag=0");

        // The runtime never calls the partial helper when the cache returned a strict or no match,
        // so the partial helper's behaviour at flag=0 is not exercised by the runtime.  The pin is:
        // the cache never returns a partial match at flag=0 -> the runtime sees a strict-or-empty
        // Match, identical to today.  T13.5 below proves the helper is invariant to the flag
        // (the flag is a cache-side concern; the helper is a pure token+shape function that fires
        // on valid partial inputs regardless of how the cache was called).
        const SuffixHookOutcome fired = suffix_reuse_partial_apply(
            {1, 2, 3, 4}, prompt, 4, 2, std::vector<CheckpointLength>{{4, 0}});
        check(fired.fired && fired.plan_edit_pos == 2 && fired.restore_anchor_length == 4,
              "T13.5: partial helper is flag-agnostic (pure token+shape function; fires on valid partial)");

        // T13.6: the helper IS fail-closed on invalid shapes (any of: L<=0, P<=0, P>=L, L>n, LCP
        // inconsistent, tokens.size()!=L).  See T15v below for the exhaustive set.
    }

    // ============================================================
    // T14: BEST-OF-N (the r2 F3/F5 selection rules)
    // ============================================================
    {
        // Three checkpoints with different LCPs against prompt {A B X Y}:
        //   ck A: {A B C D}     -> LCP = 2, depth = 4
        //   ck B: {A B C D E F} -> LCP = 2, depth = 6  (longer, same LCP - prefer deeper)
        //   ck C: {A B}         -> strict prefix (length 2) - the strict anchor, s = 2
        // r2 F5: the partial candidate (deepest donor checkpoint 6 > P = 2) BORROWS - a
        // borrowing partial is never eligible (the borrow path mounts a checkpoint at
        // exactly P and there is none).  The strict match at 2 stands.
        auto s = image({});
        s.checkpoints = {ckpt({1, 2, 3, 4}), ckpt({1, 2, 3, 4, 5, 6}), ckpt({1, 2})};
        ConversationCache cache(1 << 20, 4);
        cache.put(std::move(s));

        const std::vector<int32_t> prompt = {1, 2, 90, 91};   // [A B X Y]
        const auto m = cache.best_with_partial(64, prompt, {}, true);
        check(!m.partial && m.tokens == 2,
              "T14.1: borrowing partial (P=2 below donor's deepest 6) is skipped; strict s=2 stands (r2 F5)");
        check(m.tokens == 2,
              "T14.2: the strict anchor reads from 2 - never shallower, never stale");

        // r2 F3 cross-check: the partial is not rescued by a deep state pool either -
        // the F5 borrow skip comes first (documented choice: never borrow a partial).
        const auto m_state = cache.best_with_partial(64, prompt, {}, true, {3});
        check(!m_state.partial && m_state.tokens == 2,
              "T14.3: F5 first: even with a state anchor at 3, a borrowing partial never mounts");

        // Live-anchor shape (NOT a borrow): parked live {A B C D E F}, a strict anchor
        // {A B} (s=2), prompt {A B C X} so the live's LCP P=3 > s; a state anchor at 3
        // (<= P, > s) -> the partial wins (its deepest restorable anchor 3 beats s 2, r2 F3).
        auto s2 = image({1, 2, 3, 4, 5, 6});
        s2.checkpoints = {ckpt({1, 2})};
        ConversationCache cache2(1 << 20, 4);
        cache2.put(std::move(s2));
        const std::vector<int32_t> prompt_p3 = {1, 2, 3, 90};   // [A B C X]: live LCP 3, strict s 2
        const auto m2 = cache2.best_with_partial(64, prompt_p3, {}, true, {3});
        check(m2.partial,
              "T14.4: live-anchor partial engages when its state anchor (3) beats strict s (2)");
        check(m2.tokens == 3 && m2.checkpoint_length == 6 && m2.live,
              "T14.5: LCP=3 (> s), anchor L=6, live=true - reads never shallower than the strict 2");
        // ...and is refused without the state anchor: the donor's own periodic anchors at
        // or before P are exactly the strict pass's candidates (equal, never deeper).
        const auto m2b = cache2.best_with_partial(64, prompt_p3, {}, true);
        check(!m2b.partial && m2b.tokens == 2,
              "T14.5b: same shape, no state anchor -> strict stands (r2 F3)");

        // Longest LCP wins between two live-anchor candidates.
        auto s3 = image({});
        s3.checkpoints = {ckpt({1, 2, 3, 4}), ckpt({1, 2, 30, 31, 32})};
        ConversationCache cache3(1 << 20, 4);
        cache3.put(std::move(s3));
        const std::vector<int32_t> prompt2 = {1, 2, 3, 90, 91, 92};   // matches ck A for 3 tokens, ck B for 2
        // Both candidates are non-live partials whose donor has a deeper checkpoint ->
        // F5 skips them; the (empty) strict match stands.
        const auto m3 = cache3.best_with_partial(64, prompt2, {}, true, {3});
        check(!m3.partial && m3.tokens == 0,
              "T14.6: borrowing partials skipped regardless of LCP (r2 F5); strict 0 stands");
    }

    // ============================================================
    // T15: IMAGE / GEOMETRY MISMATCH (refuse, full re-read)
    // ============================================================
    // The checkpoint has an image at position 1; the request's images don't include it.  The
    // LCP-based image check refuses the partial mount even if the tokens match.
    {
        auto s = image({});
        s.checkpoints = {ckpt({1, 2, 3, 4})};
        s.checkpoints[0].imgs = {{1, 0xCAFE}};                   // image at position 1
        ConversationCache cache(1 << 20, 4);
        cache.put(std::move(s));

        const std::vector<int32_t> prompt = {1, 2, 90, 91};
        // strict search: image mismatch at position 1 (< 4) -> strict returns 0.
        const std::vector<ConversationImageKey> req_imgs = {{2, 0xDEAD}};  // request has an image at pos 2
        const auto strict = cache.best(prompt, req_imgs, true);
        check(strict.tokens == 0,
              "T15.1 setup: strict refuses on image mismatch at position < strict-length");

        // partial search: the LCP image check (bounded by P) must also refuse.  Here the LCP
        // could be 2 (tokens 0,1 match) but the checkpoint has an image at position 1 which is
        // below the LCP.  The request's image at position 2 is at P (not below).  So the
        // checkpoint has an image below P that the request does NOT have -> refuse.
        const auto m = cache.best_with_partial(64, prompt, req_imgs, true);
        check(!m.partial,
              "T15.2: partial refuses on image mismatch below the LCP (no fabricated mount)");

        // The partial helper itself is a pure token+shape check; the cache is the image gate.
        // In this case, the cache refuses BEFORE the helper is called.  Pin the helper's
        // token-level correctness: same shape, tokens match, helper fires.
        const SuffixHookOutcome ok = suffix_reuse_partial_apply(
            {1, 2, 3, 4}, prompt, /*L=*/4, /*P=*/2, std::vector<CheckpointLength>{});
        check(ok.fired,
              "T15.3: helper's token-level check fires (the cache is the image gate, not the helper)");
    }

    // ============================================================
    // T15 variant: PARTIAL helper fail-closed on inconsistent inputs
    // ============================================================
    {
        // L <= 0
        const SuffixHookOutcome a = suffix_reuse_partial_apply(
            {1, 2, 3, 4}, {1, 2, 90, 91}, /*L=*/0, /*P=*/2, {});
        check(!a.fired, "T15v.1: L <= 0 -> fired=false (fail-closed)");

        // P <= 0
        const SuffixHookOutcome b = suffix_reuse_partial_apply(
            {1, 2, 3, 4}, {1, 2, 90, 91}, /*L=*/4, /*P=*/0, {});
        check(!b.fired, "T15v.2: P <= 0 -> fired=false (fail-closed)");

        // P >= L (not partial, would be strict)
        const SuffixHookOutcome c = suffix_reuse_partial_apply(
            {1, 2, 3, 4}, {1, 2, 3, 4, 5}, /*L=*/4, /*P=*/4, {});
        check(!c.fired, "T15v.3: P >= L -> fired=false (not a partial shape)");

        // L > cur.size() - T17 (2026-10-13): this is now a VALID partial shape
        // (the parked-live anchor is typically longer than the next prompt).
        // The helper fires and narrows to P; the runtime restores the deepest
        // checkpoint <= P, never state@L.  The old pin refused it (three guards);
        // see T17.6 in this suite.
        const SuffixHookOutcome d = suffix_reuse_partial_apply(
            {1, 2, 3, 4, 5, 6}, {1, 2}, /*L=*/6, /*P=*/2, {});
        check(d.fired && d.plan_edit_pos == 2 && d.restore_anchor_length == 6,
              "T15v.4: L(6) > cur(2) fires (T17 live-anchor shape; anchor 6, narrow 2)");

        // Inconsistent tokens (length mismatch)
        const SuffixHookOutcome e = suffix_reuse_partial_apply(
            {1, 2, 3, 4, 5}, {1, 2, 90, 91}, /*L=*/4, /*P=*/2, {});
        check(!e.fired, "T15v.5: tokens.size() != L -> fired=false (inconsistent)");

        // LCP claim inconsistent with the actual tokens
        const SuffixHookOutcome f = suffix_reuse_partial_apply(
            {1, 2, 3, 4}, {1, 90, 3, 4}, /*L=*/4, /*P=*/2, {});   // claim P=2 but token 1 mismatches
        check(!f.fired, "T15v.6: LCP claim inconsistent with tokens -> fired=false");

        // Empty chain + valid partial = fired with empty chain (chain filter is a no-op).
        const SuffixHookOutcome g = suffix_reuse_partial_apply(
            {1, 2, 3, 4}, {1, 2, 90, 91}, /*L=*/4, /*P=*/2, {});
        check(g.fired && g.chain.empty(),
              "T15v.7: empty chain + valid partial = fired, chain stays empty");

        // Chain with the anchor + a non-anchor entry above the LCP: keep the anchor, drop the other.
        {
            std::vector<CheckpointLength> chain = { {2, 0}, {4, 1}, {6, 2} };  // L=4 is the anchor
            const SuffixHookOutcome a = suffix_reuse_partial_apply(
                {1, 2, 3, 4}, {1, 2, 90, 91}, /*L=*/4, /*P=*/2, chain);
            check(a.fired && a.chain.size() == 2 && a.chain[0].length == 2 && a.chain[1].length == 4,
                  "T15v.9: chain with anchor L=4 + non-anchor 6 > P=2 -> keep [2, 4], drop 6");
        }
        // Chain with the anchor + an entry exactly at the LCP: keep both (the LCP entry is below).
        {
            std::vector<CheckpointLength> chain = { {2, 0}, {4, 1} };
            const SuffixHookOutcome a = suffix_reuse_partial_apply(
                {1, 2, 3, 4}, {1, 2, 90, 91}, /*L=*/4, /*P=*/2, chain);
            check(a.fired && a.chain.size() == 2,
                  "T15v.10: chain with anchor L=4 + entry at P=2 -> both kept (entry length <= P)");
        }

        // Token at the divergence point matches -> LCP == L (strict, not partial).  The helper refuses
        // because the divergence check `partial_ck[P] != cur[P]` fails (they're equal -> LCP > P).
        // The strict path handles this; the partial helper must not fire.
        const SuffixHookOutcome strict = suffix_reuse_partial_apply(
            {1, 2, 3, 4}, {1, 2, 3, 90}, /*L=*/4, /*P=*/2, {});   // claim P=2 but token 2 matches (LCP=3)
        check(!strict.fired,
              "T15v.8: claim P=2 but tokens show LCP=3 -> fired=false (strict path handles)");

        // The LCP verification loop catches the inconsistency: a P=2 claim on tokens where the
        // first 2 match but the 3rd also matches means the caller mis-computed P.  The helper's
        // LCP verification walks [0, P) only; it does NOT walk beyond P (the divergence check at
        // [P] is the next guard).  When P==L (which the L>=P guard catches above), it's strict.
        // For P < L but token[P] == cur[P], the LCP would be > P, not == P - the divergence
        // check catches it: the helper returns fired=false.
    }

    // ============================================================
    // T16: SPLICE-ENGAGEMENT REVISITED (the r2 F3 dominance rule)
    // ============================================================
    // The M1 steady-state finding (2026-10-13) made best_with_partial prefer a
    // deeper partial over a shallower strict anchor on "L > s" alone.  The r2 review
    // (finding 3) showed that with the splice gone this can pick a WORSE restore
    // point than flag-off: reads start at the restored anchor, and a donor's
    // periodic checkpoints at or before P are prefixes the strict pass already
    // sees.  The rule now: the partial is chosen only when its deepest restorable
    // anchor at or before P - across the donor's periodic checks AND the session's
    // state pool - is DEEPER than the strict match's s.  Otherwise the strict
    // mount wins (never shallower than flag-off).
    {
        // One conversation: checkpoints {A B} (2, the strict anchor) and
        // {A B C D E} (5, crosses the edit).  Request {A B C X E}: history
        // diverges at D=3 (C vs X at index 3).  Strict anchor {A B} mounts 2;
        // partial on the live {A B C D E}: LCP=3, L=5.
        auto s = image({1, 2, 3, 4, 5});
        s.checkpoints = { ckpt({1, 2}), ckpt({1, 2, 3, 4, 5}) };
        ConversationCache cache(1 << 20, 4);
        cache.put(std::move(s));

        const std::vector<int32_t> prompt = {1, 2, 3, 90, 5};   // edits token 3 (X=90)
        const auto strict = cache.best(prompt, {}, true);
        check(strict.tokens == 2 && !strict.partial,
              "T16.1: strict best() mounts the {A B} anchor (2)");

        // Without a state anchor the partial is NOT chosen: its deepest restorable
        // anchor <= P=3 is the donor's {A B} (2) - exactly the strict anchor.  Equal,
        // never deeper: the partial would restore at 2 like the strict mount, at the
        // cost of taking and restoring the whole donor image.
        const auto m0state = cache.best_with_partial(64, prompt, {}, true);
        check(!m0state.partial && m0state.tokens == 2,
              "T16.2: no deeper anchor -> strict stands (r2 F3: the partial must earn its keep)");

        // With a state-pool checkpoint at 3 (<= P, > s) the partial wins: it restores
        // at 3 where the strict mount restores at 2.
        const auto m = cache.best_with_partial(64, prompt, {}, true, {3});
        check(m.partial,
              "T16.3: partial chosen when its state anchor (3) is deeper than strict s (2)");
        check(m.tokens == 3,
              "T16.4: partial LCP == 3 (> s); the restore point (3) is deeper than the strict 2");
        check(m.checkpoint_length == 5,
              "T16.5: the deeper checkpoint (5) is the anchor");

        // The neutral case: strict anchor AT the LCP depth (s == p == 3).  The
        // donor's own anchors at or below P never beat s (they are the strict
        // pass's candidates), so only a state anchor > 3 could win - there is
        // none at or below P=3.
        auto s2 = image({1, 2, 3, 4, 5});
        s2.checkpoints = { ckpt({1, 2, 3}), ckpt({1, 2, 3, 4, 5}) };
        ConversationCache cache2(1 << 20, 4);
        cache2.put(std::move(s2));
        const auto m2 = cache2.best_with_partial(64, prompt, {}, true);
        check(!m2.partial && m2.tokens == 3,
              "T16.6: s == p -> strict stands (the donor's periodic anchors never beat it)");

        // Off-flag: byte-identical strict (the early-return never changes behavior).
        const auto m0 = cache.best_with_partial(0, prompt, {}, true);
        check(!m0.partial && m0.tokens == 2,
              "T16.7: flag=0 stays byte-identical strict (2)");

        // Cross-conversation (review r1 finding 1 + r2 F3's repro): conversation A
        // offers a strict anchor at 5; conversation B shares 6 tokens (LCP 6 > s!)
        // but its only checkpoint at or before 6 is at 1.  The partial must NOT be
        // taken: mounting it would restore at 1 where flag-off restores at 5.
        {
            auto convA = image({1, 2, 3, 4, 5, 50, 51});
            convA.checkpoints = { ckpt({1, 2, 3, 4, 5}) };
            auto convB = image({1, 2, 3, 4, 5, 6, 70, 71, 72, 73});
            convB.checkpoints = { ckpt({1}) };
            ConversationCache cache3(1 << 20, 4);
            cache3.put(std::move(convA));
            cache3.put(std::move(convB));

            const std::vector<int32_t> p2 = {1, 2, 3, 4, 5, 6, 80, 81};
            const auto m3 = cache3.best_with_partial(64, p2, {}, true);
            check(!m3.partial && m3.tokens == 5,
                  "T16.8: cross-conversation partial with a shallower anchor (1 < s=5) never mounts (r2 F3)");
            check(m3.tokens == 5 && !m3.partial,
                  "T16.8b: strict anchor (5) stands over B's deeper-LCP but anchor-poor partial");

            // r4 F1 (review r3 MAJOR 1): a state entry from an UNRELATED conversation is
            // not a restorable anchor.  generate.cpp builds sfx_state_lengths with the same
            // starts_with rule the post-mount off_path erase uses, so the entry at 6 (whose
            // ids are NOT a prefix of p2) never reaches this call.  Pinned in two steps:
            // the RAW pool {6} WOULD flip the decision (the hazard the caller-side filter
            // closes - this is the reviewer's xconv.cpp repro), and the FILTERED pool
            // (empty here) keeps the strict mount at 5, exactly as T16.8.
            const auto m_raw = cache3.best_with_partial(64, p2, {}, true, {6});
            check(m_raw.partial && m_raw.tokens == 6 && m_raw.checkpoint_length == 10,
                  "T16.8c (hazard): the RAW unfiltered pool {6} WOULD take B's partial (tokens=6, L=10)");
            const auto m3c = cache3.best_with_partial(64, p2, {}, true);
            check(!m3c.partial && m3c.tokens == 5,
                  "T16.8c: a NON-prefix state entry at 6 does not change the decision (strict mount at 5 wins)");
        }

        // ============================================================
        // T16.9 (r4 F2, review r3 MAJOR 2): the anchor must beat the SESSION's
        // floor - max(local resume, slot_tokens) - not just the parked strict s.
        // The reviewer's numbers: session resume = 100 (the floor), the parked
        // strict match is 64, the donor's LCP is P = 200 with a prefix state
        // anchor at 90 (or 150).
        // ============================================================
        {
            std::vector<int32_t> prompt;
            for (int32_t i = 0; i < 256; ++i) prompt.push_back(i + 1);
            // Conversation A: live + checkpoint are a strict 64-token prefix of the prompt.
            auto convA = image({1});
            convA.live.ids.assign(prompt.begin(), prompt.begin() + 64);
            ConversationCheckpoint c64;
            c64.ids.assign(prompt.begin(), prompt.begin() + 64);
            c64.gdn.resize(64, 1);
            convA.checkpoints = { c64 };
            // Conversation B (the donor): live shares 200 tokens then diverges; its only
            // periodic checkpoint is at 1, so the state pool alone supplies the anchor.
            auto convB = image({1});
            convB.live.ids.assign(prompt.begin(), prompt.begin() + 200);
            for (int32_t t = 5001; t <= 5003; ++t) convB.live.ids.push_back(t);   // L = 203
            convB.checkpoints = { ckpt({1}) };
            ConversationCache cache4(1 << 20, 4);
            cache4.put(std::move(convA));
            cache4.put(std::move(convB));

            const auto strict = cache4.best(prompt, {}, true);
            check(strict.tokens == 64 && !strict.partial,
                  "T16.9.1: setup - the parked strict match is A's 64-token prefix");

            // r3 bug shape: without the floor, the anchor at 90 beats s=64 and the
            // partial is taken even though the session's own resume is 100 - the
            // restore would land at 90, SHALLOWER than flag-off's 100.
            const auto m_bug = cache4.best_with_partial(64, prompt, {}, true, {90});
            check(m_bug.partial && m_bug.tokens == 200,
                  "T16.9.2 (hazard): without the floor, anchor 90 > s=64 takes the partial (the r3 MAJOR 2 shape)");

            // With the floor = max(resume, slot_tokens) = 100: anchor 90 loses, strict wins.
            const auto m_floor = cache4.best_with_partial(64, prompt, {}, true, {90}, 100);
            check(!m_floor.partial && m_floor.tokens == 64,
                  "T16.9.3: anchor 90 vs floor 100 -> strict wins (resume=100, anchor=90)");

            // Anchor 150 beats the floor 100 -> the partial wins and restores at 150.
            const auto m_deep = cache4.best_with_partial(64, prompt, {}, true, {150}, 100);
            check(m_deep.partial && m_deep.tokens == 200 && m_deep.checkpoint_length == 203,
                  "T16.9.4: anchor 150 vs floor 100 -> partial wins (P=200, L=203)");
        }
    }

    // ============================================================
    // T17: LIVE-ANCHOR PARTIAL - a parked live LONGER than the prompt
    // ============================================================
    // The M1 probe finding (2026-10-13, chain /tmp/dt-steady4.log, C-turn):
    //   parked-scan: e0.live:L=5786:skip   (prompt_len=5736)
    // The parked LIVE (prompt+response of the previous turn) is almost always
    // slightly LONGER than the next turn's prompt, and the old guards refused it
    // without computing the LCP.  The live anchor remains the most valuable partial
    // CANDIDATE - but under the r2 F3 rule it is only CHOSEN when its deepest
    // restorable anchor at or before P beats the strict s (a live partial never
    // borrows, r2 F5, so it is the one partial shape that can still win).
    //
    // Safety: the LCP walk stops at min(L, prompt.size()) - tokens past the
    // prompt are irrelevant (the partial mount discards them); the strict pass
    // keeps its own guard (a longer-than-prompt checkpoint can NEVER be a strict
    // prefix).  The runtime routes a live-anchor partial through the
    // narrowed_resume_target_state restore (state@L is stale for the edited
    // history): restore the deepest checkpoint <= P across both pools, arena
    // covers [ckpt, P) by the LCP, read [restore_length, n).
    {
        // Parked live {A B C D E} (5) + checkpoints {A B}(2), {A B C}(3).
        // Request {A B C X} (4) - SHORTER than the live, edits index 3.
        // Strict: {A B C} mounts 3 (live and {A B C D E} fail: longer/edited).
        // Partial candidate: the live, LCP = 3, L = 5.
        auto s = image({1, 2, 3, 4, 5});
        s.checkpoints = { ckpt({1, 2}), ckpt({1, 2, 3}) };
        ConversationCache cache(1 << 20, 4);
        cache.put(std::move(s));

        const std::vector<int32_t> prompt = {1, 2, 3, 90};   // len 4 < live 5
        const auto strict = cache.best(prompt, {}, true);
        check(strict.tokens == 3 && !strict.partial,
              "T17.1: strict best() mounts {A B C} (3); the longer live is refused");

        // r2 F3: the live-anchor partial's deepest restorable anchor <= P=3 is the
        // donor's {A B C} (3) == s - equal, never deeper - so the partial is NOT
        // chosen.  Same-conversation partials can never beat the strict mount via
        // periodic checkpoints (the strict pass sees those); only a state-pool
        // checkpoint deeper than s wins.
        const auto m_eq = cache.best_with_partial(64, prompt, {}, true);
        check(!m_eq.partial && m_eq.tokens == 3,
              "T17.2: s == deepest anchor -> strict stands (the partial must earn its keep, r2 F3)");
        const auto m_st = cache.best_with_partial(64, prompt, {}, true, {3});
        check(!m_st.partial && m_st.tokens == 3,
              "T17.3: a state anchor at s (3, not > s) does not rescue the partial either");

        // With only {A B} parked (s=2) and a state checkpoint at 3 (<= P=3, > s):
        // the live-anchor partial WINS - it restores at 3 where strict restores at 2.
        auto s2 = image({1, 2, 3, 4, 5});
        s2.checkpoints = { ckpt({1, 2}) };
        ConversationCache cache2(1 << 20, 4);
        cache2.put(std::move(s2));
        const auto m = cache2.best_with_partial(64, prompt, {}, true, {3});
        check(m.partial,
              "T17.4: live anchor chosen when its state anchor (3) beats strict s (2)");
        check(m.tokens == 3,
              "T17.5: live-anchor partial LCP == 3 (divergence at index 3)");
        check(m.checkpoint_length == 5,
              "T17.6: the parked LIVE (5) is the anchor");
        check(m.live,
              "T17.7: Match.live == true (the anchor is the parked live)");

        // The helper accepts L > n_new for the partial path (fail-closed kept
        // for every OTHER shape).  Fire with L=5, P=3, cur len 4: plan narrows
        // to P=3, anchor 5.
        const std::vector<int32_t> live_tokens = {1, 2, 3, 4, 5};
        std::vector<CheckpointLength> chain = { {2, 0}, {3, 1} };
        const SuffixHookOutcome outcome = suffix_reuse_partial_apply(
            live_tokens, prompt, /*L=*/5, /*P=*/3, chain);
        check(outcome.fired,
              "T17.8: partial_apply fires with L(5) > n_new(4)");
        check(outcome.fired && outcome.plan_edit_pos == 3,
              "T17.9: plan_edit_pos == 3 (the LCP)");
        check(outcome.fired && outcome.restore_anchor_length == 5,
              "T17.10: restore_anchor_length == 5 (the live anchor)");
        check(compute_suffix_plan(live_tokens, prompt).old_suffix_len == 0,
              "T17.11: plan.old_suffix_len == 0 ({D E} vs {X} - no common tail; on the plan, r2 F7)");

        // conversation_lcp direct: live longer than prompt walks to the prompt's
        // end and reports the prompt-bounded LCP (not 0).
        ConversationCheckpoint lc;
        lc.ids = {1, 2, 3, 4, 5};
        check(strata::core::conversation_lcp(lc, prompt, {}) == 3,
              "T17.12: conversation_lcp walks a longer-than-prompt checkpoint (3, not 0)");

        // Off-flag: byte-identical strict - the live-anchor lift is flag-gated.
        const auto m0 = cache2.best_with_partial(0, prompt, {}, true);
        check(!m0.partial && m0.tokens == 2,
              "T17.13: flag=0 stays byte-identical strict (2)");

        // A strict-longer guard REMAINS: a candidate longer than the prompt can
        // never produce p >= L (strict); the partial pass alone evaluates it.
        // Pin that conversation_lcp reports the FULL checkpoint length when the
        // checkpoint is a strict prefix (shorter or equal) - unchanged.
        ConversationCheckpoint sc;
        sc.ids = {1, 2};
        check(strata::core::conversation_lcp(sc, prompt, {}) == 2,
              "T17.14: shorter strict-prefix checkpoint still reports its full length (2)");
    }

    // ============================================================
    // T18: ALWAYS-FAIL-CLOSED GAP READ (dt-steady5 measured finding)
    // ============================================================
    // Reading [P, n) on state@checkpoint loses the gap [ckpt, P) from the
    // GDN/PLE recurrence: attention sees the gap rows through valid arena K/V,
    // the running state never ingested them (C under-counted the ledger; gap 826
    // tokens).  The replay gate that once re-enabled the gap-read never shipped - the
    // splice layer it belonged to was removed after review.  The helper itself is
    // fail-closed (r2 F4): read_from == restore_length ALWAYS - the REAL helper
    // `narrowed_resume_target` (the same one generate.cpp calls), not a local copy.
    {
        using strata::program::suffix_reuse::NarrowedResumeTarget;
        using strata::program::suffix_reuse::narrowed_resume_target;
        const std::vector<int64_t> chain = { 1574, 2598 };
        // T18.1 calls the REAL helper and asserts read_from equals the length the
        // restored state describes (the reviewer's round-1 assert, de-tautologized).
        const NarrowedResumeTarget t1 = narrowed_resume_target(3424, chain);
        check(t1.restore_length == 2598,
              "T18.1a: the deepest checkpoint <= P (3424) is 2598");
        check(t1.read_from == t1.restore_length && t1.read_from == 2598,
              "T18.1b: read_from == restore_length (2598) - the prefill re-reads [2598, n), never [P, n)");
        check(narrowed_resume_target(3424, {}).read_from == 0,
              "T18.3: no checkpoint <= P -> fresh read from 0 (fail-closed, never fabricate)");
        check(narrowed_resume_target(2000, chain).restore_length == 1574,
              "T18.4: P between anchors -> the deeper-below-P anchor (1574)");
        {
            const auto t = narrowed_resume_target(3424, chain);
            check(t.read_from == 2598 && t.restore_length == 2598,
                  "T18.2: the helper's raw read_from is restore_length (2598), not P - "
                  "byte-equal to strict-path semantics by construction");
        }
    }

    // ============================================================
    // T22 (FW-2b): state-cadence checkpoints shrink the fail-close gap.
    // checkpoint_save copies state only (gdn/ple/tails/dead/block_pos; no K/V),
    // so fine-cadence state checkpoints are cheap (T19: ~100-150ms each).  They
    // live in a SIDE pool (separate from the LRU-capped `checks` list) so they
    // never evict the deep periodic anchors.  The restore pick must take the
    // deepest checkpoint <= P across BOTH pools; fail-close semantics unchanged
    // (the gap read is the deepest anchor below P; always fail-closed).
    // Pure seams under test: `narrowed_resume_target_state` (the two-pool pick)
    // and `state_cadence_next` (the fine-cadence ladder, never past n).
    {
        using strata::program::suffix_reuse::narrowed_resume_target_state;
        using strata::program::suffix_reuse::state_cadence_next;
        using strata::program::suffix_reuse::NarrowedResumeTarget;

        // T22.1: the deep-pool checkpoint wins when it is the deepest <= P.
        // checks = {1574, 2598} (periodic, prompt_cache LRU), state = {2816, 3072}
        // (fine cadence).  P = 3424: deepest across both = 3072 (not 2598).
        {
            const std::vector<int64_t> checks = { 1574, 2598 };
            const std::vector<int64_t> state  = { 2816, 3072 };
            const NarrowedResumeTarget t = narrowed_resume_target_state(3424, checks, state);
            check(t.restore_length == 3072,
                  "T22.1: state-cadence 3072 beats periodic 2598 (deepest across both pools)");
            check(t.read_from == 3072,
                  "T22.1b: read starts at the restored checkpoint (3072; read_from == restore_length, fail-closed)");
        }

        // T22.2: the periodic pool wins when it is deeper (state pool empty).
        {
            const NarrowedResumeTarget t = narrowed_resume_target_state(3424, { 1574, 2598 }, {});
            check(t.restore_length == 2598 && t.read_from == 2598,
                  "T22.2: no state checkpoints -> deepest periodic (2598), read_from == restore_length");
        }

        // T22.3: fail-closed degenerates.  P below every checkpoint in both pools
        // -> fresh read from 0; nothing is fabricated.
        {
            const NarrowedResumeTarget t = narrowed_resume_target_state(1000, { 1574 }, { 2816 });
            check(t.restore_length == 0 && t.read_from == 0,
                  "T22.3: P below both pools -> fresh read from 0 (fail-closed, never fabricate)");
        }

        // T22.4: equal lengths are fine (dedup not required; deepest is deepest).
        {
            const NarrowedResumeTarget t = narrowed_resume_target_state(3424, { 2598, 3424 }, { 3424 });
            check(t.restore_length == 3424 && t.read_from == 3424,
                  "T22.4: equal lengths across pools -> the shared depth (3424)");
        }

        // T22.5: the fine-cadence ladder.  state_cadence_next(done, n, every)
        // returns the next capture position: done + every, clamped so a capture
        // never lands past the prompt end (a checkpoint at exactly n is the live
        // conversation, not a checkpoint).
        check(state_cadence_next(2598, 5736, 256) == 2854,
              "T22.5: ladder from 2598 at 256 cadence -> 2854");
        check(state_cadence_next(5600, 5736, 256) == 5736,
              "T22.5b: clamped to n when done+every overshoots (never past the prompt)");
        check(state_cadence_next(5600, 5736, 0) == INT64_MAX,
              "T22.5c: every = 0 -> off (INT64_MAX, the gate never fires)");
    }

    // ============================================================
    // T23 (r2 F2): the regenerate / identical-retry shape - p == prompt.size()
    // ============================================================
    // Reviewer repro (r2 finding 2): parked live of length 9 (prompt 6 + reply 3),
    // a prompt of length 6 that is a PREFIX of it, checkpoints at 2 and 5.
    // conversation_lcp walks to p == 6 == prompt.size(); taking that as a partial
    // match would take the donor whole with resume == n and from_live == true -
    // the partial branch (resume < cur.size()) and the restore branch (from_live)
    // are both skipped, the session sits at the stale state after the reply
    // tokens, and the last prompt token never starts a verify window.  The
    // non-live variant fails with "ERR restoring a conversation checkpoint
    // failed".  The bound p < prompt.size() (upstream's strict
    // conversation_prefix applies the same n < prompt.size()) refuses it: the
    // match must behave exactly like the strict mount (resume 5), never
    // resume == 6-with-stale-state, never an error path.
    {
        // parked conversation: live = prompt(6) + reply(3); checkpoints at 2 and 5
        // (prefixes of the prompt).
        auto s = image({1, 2, 3, 4, 5, 6, 7, 8, 9});
        s.checkpoints = {ckpt({1, 2}), ckpt({1, 2, 3, 4, 5})};
        ConversationCache cache(1 << 20, 4);
        cache.put(std::move(s));
        const std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6};   // regenerate: a prefix of the parked live

        const auto st = cache.best(prompt, {}, true);
        const auto m = cache.best_with_partial(64, prompt, {}, true);
        check(st.tokens == 5,
              "T23.1: strict best() mounts the checkpoint at 5 (the deepest prefix)");
        check(!m.partial && m.tokens == 5,
              "T23.2: best_with_partial behaves EXACTLY like the strict mount (5), never tokens=6 partial");
        check(m.tokens < (int64_t) prompt.size(),
              "T23.3: resume (5) < n (6) - the last prompt token still starts a verify window");

        // Even a deep state pool cannot rescue a p == n candidate: it never forms.
        const auto m_state = cache.best_with_partial(64, prompt, {}, true, {5});
        check(!m_state.partial && m_state.tokens == 5,
              "T23.4: a state anchor does not make the p == n shape eligible (bound first)");

        // The non-live variant (the r2 "ERR restoring" repro): a parked CHECKPOINT
        // longer than the prompt whose tokens all match - strict refuses it, the
        // partial pass must refuse it too, leaving the no-match (fresh prefill)
        // path - never a taken-whole donor with no mountable anchor.
        auto s2 = image({42, 43});
        s2.checkpoints = {ckpt({1, 2, 3, 4, 5, 6, 7, 8, 9})};
        ConversationCache cache2(1 << 20, 4);
        cache2.put(std::move(s2));
        const auto m2 = cache2.best_with_partial(64, prompt, {}, true);
        check(!m2.partial && m2.tokens == 0,
              "T23.5: non-live p == n candidate refused - no match, no error shape (r2 F2)");
    }

    std::printf("suffix_reuse_partial_prefix_test: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
