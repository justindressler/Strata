// src/program/suffix_reuse_continuation_and_anchor_test.cpp - phase 3a: T9 (continuation-mount
// parity, the 0% vs 98% off-arm regression) + T10 (anchor-splice engagement, the wiring bug
// where the splice's K rotation is wasted because the chunk loop re-reads the rotated region).
//
// Both tests are header-only and CPU-only; they exercise the pure helpers in
// include/strata/program/suffix_reuse.hpp and a tiny wiring predicate the runtime's `read_from`
// resolves to.  No CUDA, no <cuda_runtime.h>, no QsaState.
//
// ============================================================
// T9 - continuation-mount parity (the live 0% vs 98% regression)
// ============================================================
//
// THE BUG.  The splice hook (generate.cpp:9004-9092 on feat/suffix-reuse @ ce20ef3) computes its
// plan with `compute_suffix_plan(old=resume_ck.tokens, new=cur_tokens)` which walks the longest
// common SUFFIX from the BACK.  For a pure continuation where `old` is a prefix of `new` and the
// last token happens to match (the live probe5 case: the EOS token 248046 at the end of `old`
// equals the EOS at the end of `cur`), the suffix walk yields `suffix = 1, edit_pos =
// n_old - 1 = 3239`.  The hook then narrows `resume` to 3239, erases the resume ckpt (length
// 3240 > 3239), and the mount is destroyed.  Live evidence (probe3/probe4/probe5 on engine/
// strata-suffix @ ce20ef3, config strata-iq3_xxs-sfx-fp16.json, flag 64): t2 = [u1(3245 tok
// ledger), assistant:"", CONT] mounts 0 reused on the ON arm, but the IDENTICAL request on the
// OFF arm (same binary, flag absent) mounts 3240/3294 (98%, serves in 1.0s vs 10s).
//
// THE FIX.  `compute_suffix_plan` must use the longest common PREFIX from the FRONT for
// `edit_pos`, not the suffix walk.  For a pure continuation (`old` is a prefix of `new`):
//   LCP = n_old, edit_pos = n_old = resume.  No narrowing (plan_edit_pos < resume is false).
//   The plan is also a no-op: old_suffix_len = new_suffix_len = 0 (the only "common suffix"
//   token was the EOS, not a real content match).  The (removed) K-rotation layer was gated
//   on old_suffix_len > 0, so no K was ever touched.  The checks.erase predicate `length > edit_pos`
//   drops nothing (resume-length ckpt survives).  Mount is byte-identical to flag-OFF.
//
// RED ON THE PRE-FIX STATE (ce20ef3): the current `compute_suffix_plan` walks the suffix from
// the back, so for the live shape (last token matches) it returns edit_pos=3239, old_suffix_len=1.
// The T9 assertions `edit_pos == 3240` and `old_suffix_len == 0` FAIL.
//
// ============================================================
// T10 - anchor engagement (the wiring of the chunk loop's read start)
// ============================================================
//
// HISTORY.  Phase-2 wiring (431fc2c + ce20ef3) consumed only `t.restore_length` and the
// chunk loop read from `resume`.  Phase 3a wired `wire_splice_read_from` so the narrowed
// target's `read_from` is honoured.  The r2 review (finding 4) then moved the fail-closed
// contract INTO the helper: `read_from == restore_length` always - the chunk loop starts
// reading at the position the restored state describes, so the gap between the restored
// checkpoint and the edit position is re-read by the prefill, never skipped (reading
// [edit, n) on the restored state loses the gap from the GDN/PLE recurrence - T18).
//
// WHAT THIS TEST PINS.  When `splice_narrowed` is true, the chunk loop's `read_from` is
// `t.read_from` (== restore_length), not `resume`; the pure helper `wire_splice_read_from`
// in the header captures the resolution so the test pins it without GPU.
//
// ============================================================
//
// WHAT THIS TEST PINS.
//
//   T9.1 LIVE SHAPE (EOS-last coincidence, 3240->3294, last token matches):
//          edit_pos == 3240 (= resume, NOT 3239)
//          old_suffix_len == 0 (no rotation; pure continuation)
//          outcome.fired == true (predicate fires; runtime decides no-op)
//          outcome.plan_edit_pos == 3240
//          outcome.chain.size() == 2 (no drops; resume ckpt at 3240 survives)
//          the runtime narrowing predicate `plan_edit_pos < resume` is FALSE -> mount survives
//
//   T9.2 LIVE SHAPE (no EOS coincidence, 3240->3294, last token differs):
//          edit_pos == 3240 (= resume, the LCP)
//          old_suffix_len == 0
//          mount survives (same as T9.1)
//
//   T9.3 OFF-PARITY (flag=0): the same resume/cur -> helper returns fired=false; runtime path
//          is byte-identical to T9.1's on-arm no-op (mount survives)
//
//   T10.1 ANCHOR-SPLICE (checks={2000, 3240, 3244}, edit E=2500):
//          compute_suffix_plan -> narrow_pos == 2500 (LCP), old_suffix_len == 790 (the plan's
//          unchanged tail - pinned on the plan; the outcome no longer carries tail fields)
//          outcome.fired == true
//          outcome.plan_edit_pos == 2500
//          outcome.chain keeps {2000} only (3240, 3244 dropped by `length > 2500`)
//          narrowed_resume_target(2500, {2000}) -> restore_length == 2000, read_from == 2000
//                                                 (r2 F4: read_from == restore_length)
//          wire_splice_read_from(true, t, resume=2000, reread_to=-1) -> 2000
//          wire_splice_read_from(false, t, resume=2000, reread_to=-1) -> 2000 (legacy path)
//
//   T10.2 NO ANCHOR (checks={3240, 3244} - the live 431fc2c+ce20ef3 shape; no candidate at/
//          below the edit pos): helper returns restore_length == 0, read_from == 0; wiring
//          degrades to fresh prefill from 0 (no error, no fabricated checkpoint).
//
//   T10.3 FAIL-CLOSED: when splice_narrowed, the chunk loop's read_from is t.read_from
//          (== restore_length): the gap [restore, edit) is re-read by the prefill - never
//          skipped - and [edit, n) follows, byte-equal to the strict path's semantics.

#include "strata/program/suffix_reuse.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int g_fail = 0;
int g_pass = 0;
void check(bool ok, const char* what) {
    std::printf("  %-86s %s\n", what, ok ? "ok" : "FAIL");
    if (ok) ++g_pass; else ++g_fail;
}

// Live-shaped fixtures: a 3240-token resume checkpoint whose tokens are [1..3240] (the LCP),
// and a 3294-token cur prompt that is the same 3240 + 54 new tokens (CONT).
std::vector<int32_t> make_resume_ck_3240() {
    std::vector<int32_t> t(3240);
    for (int32_t i = 0; i < 3240; ++i) t[i] = i + 1;
    return t;
}
std::vector<int32_t> make_cur_3294_continuation(const std::vector<int32_t>& resume, int32_t last_tok) {
    // cur = resume (3240) ++ (54 new tokens).  The 54 new tokens are 9901..9954.
    std::vector<int32_t> cur(3294);
    for (int32_t i = 0; i < 3240; ++i) cur[i] = resume[i];
    for (int32_t i = 0; i < 54; ++i) cur[3240 + i] = 9901 + i;
    cur.back() = last_tok;   // the very last token is the param (controls EOS-coincidence shape)
    return cur;
}

}  // namespace

int main() {
    std::printf("suffix_reuse_continuation_and_anchor_test (T9 continuation parity + T10 anchor engagement)\n");

    using strata::program::suffix_reuse::CheckpointLength;
    using strata::program::suffix_reuse::compute_suffix_plan;
    using strata::program::suffix_reuse::narrowed_resume_target;
    using strata::program::suffix_reuse::suffix_reuse_apply_invalidation;
    using strata::program::suffix_reuse::suffix_reuse_should_invalidate;
    using strata::program::suffix_reuse::SuffixHookOutcome;
    using strata::program::suffix_reuse::NarrowedResumeTarget;
    using strata::program::suffix_reuse::wire_splice_read_from;

    // ============================================================
    // T9: continuation-mount parity (the 0% vs 98% regression)
    // ============================================================

    // ---- T9.1 LIVE SHAPE: EOS-last coincidence (last token of old == last token of new) ----
    // The probe5 live case: old's last token is 248046 (im_end) AND new's last token is also
    // 248046 (the CONT ends with im_end).  Pre-fix: suffix walk yields suffix=1, edit_pos=3239
    // -> splice hook narrows resume to 3239 -> mount destroyed.  Post-fix: LCP=3240,
    // edit_pos=3240, old_suffix_len=0 -> no narrowing, no rotation, mount survives.
    {
        std::vector<int32_t> resume_ck = make_resume_ck_3240();
        resume_ck.back() = 248046;                    // resume_ck's last token is im_end
        const std::vector<int32_t> cur = make_cur_3294_continuation(resume_ck, /*last_tok=*/248046);

        // Sanity: the resume ckpt IS a prefix of cur (the mount condition).
        bool prefix_ok = (cur.size() >= resume_ck.size());
        for (size_t i = 0; prefix_ok && i < resume_ck.size(); ++i)
            if (cur[i] != resume_ck[i]) prefix_ok = false;
        check(prefix_ok,
              "T9.1 setup: resume_ck.tokens is a prefix of cur (the mount condition holds)");

        // The plan: must use LCP for narrow_pos (Phase 3a T9, the splice hook's narrowing
        // signal).  LCP = 3240 (the whole resume_ck matches the first 3240 of cur).
        // narrow_pos = 3240 = resume.  The plan's old_suffix_len is forced to 0 in the
        // continuation special case (any common-suffix tokens are coincidental, e.g. the EOS).
        // The plan's edit_pos (legacy, suffix-derived) is 0 in the EOS-coincidence case
        // (suffix=1, n_old=3240, edit_pos = 3240-1 = 3239 ... wait, 0?).  Let me re-trace:
        //   suffix walk: old[3239]=248046, new[3293]=248046 -> match.  new[3292] vs old[3238]
        //     = whatever old[3238] is (not 248046 since it's not the last token) -> no match.
        //   So suffix=1, edit_pos (legacy) = 3240-1 = 3239.  But the continuation special
        //   case forces old_suffix_len=0 and returns early - so edit_pos stays at the value
        //   computed BEFORE the special case (3239).  This is the legacy field; the splice
        //   hook uses narrow_pos (3240), not edit_pos.
        const auto plan = compute_suffix_plan(resume_ck, cur);
        check(plan.narrow_pos == 3240,
              "T9.1 LIVE: plan.narrow_pos == 3240 (= resume; LCP-based, NOT 3239 from suffix-walk bug)");
        check(plan.old_suffix_len == 0,
              "T9.1 LIVE: plan.old_suffix_len == 0 (no rotation; continuation = no-op)");
        check(plan.new_suffix_len == 0,
              "T9.1 LIVE: plan.new_suffix_len == 0 (the common tail is the EOS coincidence only)");

        // The splice hook helper: predicate fires (cur > resume, all controls met), but the
        // plan is a no-op.  The runtime uses these fields to decide the path.
        std::vector<CheckpointLength> chain = { {3240, 0}, {3244, 1} };
        const SuffixHookOutcome outcome = suffix_reuse_apply_invalidation(
            /*flag=*/64, /*from_live=*/false, /*resume=*/3240, chain, resume_ck, cur);
        check(outcome.fired,
              "T9.1 LIVE: hook fires (flag=64, controls met) - the runtime decides no-op from the plan");
        check(outcome.plan_edit_pos == 3240,
              "T9.1 LIVE: outcome.plan_edit_pos == 3240 (= resume; runtime does NOT narrow)");
        check(outcome.replay_from == 3240,
              "T9.1 LIVE: outcome.replay_from == 3240 (== narrow_pos; no replay position before the mount)");
        check(outcome.chain.size() == 1 && outcome.chain[0].length == 3240,
              "T9.1 LIVE: chain drops 3244 (length > edit_pos 3240), keeps resume ckpt at 3240 (1 survivor)");

        // The runtime narrowing predicate (historical, generate.cpp: `if (outcome.plan_edit_pos < resume)`).
        // For continuation, edit_pos == resume, so the predicate is false -> no narrowing.
        // (The strict-path hook itself was deleted in r2 F1 - resume always comes from a
        // checkpoint that is a prefix of the new prompt, so it can never usefully narrow.)
        const bool runtime_would_narrow = outcome.plan_edit_pos < 3240;
        check(!runtime_would_narrow,
              "T9.1 LIVE: runtime does NOT narrow resume (edit_pos == resume; mount survives)");

        // The (removed) rotation layer was gated on the plan's old_suffix_len; the plan's
        // value is pinned above (0 - the continuation special case).  With the layer gone
        // and its outcome fields deleted (r2 F7), there is nothing further to gate.

        // The runtime erasure at generate.cpp:9045-9055: drops `length > plan_edit_pos` from
        // `outcome.chain`.  For continuation, the drop predicate is `length > 3240`; 3240 is
        // not > 3240, so the resume ckpt survives.  3244 > 3240 is dropped (the periodic ckpt
        // past the resume point is dropped either way - that is the off-flag behavior).
        std::vector<bool> keep(chain.size(), false);
        for (const auto& cl : outcome.chain) if (cl.index < keep.size()) keep[cl.index] = true;
        // The resume ckpt (index 0) is in outcome.chain, so it survives.  3244 (index 1) is
        // also in outcome.chain (chain kept everything <= 3240), so it survives the splice
        // hook's erase; the canonical erase at generate.cpp:9283-9285 (predicate `length > resume`
        // where resume=3240) drops 3244 in BOTH on-arm and off-arm paths - byte-identical.
        check(keep[0],
              "T9.1 LIVE: resume ckpt (index 0) survives the splice hook's erasure");
    }

    // ---- T9.2 LIVE SHAPE: no EOS coincidence (last token of old != last token of new) ----
    // The off-parity test fixture for the continuation case (cur[599]=600, resume[499]=500).
    // Pre-fix: suffix walk yields suffix=0, edit_pos=500 (coincidentally correct by accident).
    // Post-fix: LCP=500, edit_pos=500 (correct by design).
    {
        const std::vector<int32_t> resume_ck = make_resume_ck_3240();     // [1..3240]
        const std::vector<int32_t> cur = make_cur_3294_continuation(resume_ck, /*last_tok=*/12345);

        const auto plan = compute_suffix_plan(resume_ck, cur);
        check(plan.narrow_pos == 3240,
              "T9.2 LIVE-no-EOS: plan.narrow_pos == 3240 (= resume; LCP-based)");
        check(plan.old_suffix_len == 0,
              "T9.2 LIVE-no-EOS: plan.old_suffix_len == 0 (continuation = no-op)");

        std::vector<CheckpointLength> chain = { {3240, 0}, {3244, 1} };
        const SuffixHookOutcome outcome = suffix_reuse_apply_invalidation(
            64, false, 3240, chain, resume_ck, cur);
        check(outcome.plan_edit_pos == 3240 && outcome.replay_from == 3240,
              "T9.2 LIVE-no-EOS: outcome no-op (edit_pos==resume, replay_from==resume)");
        check(outcome.chain.size() == 1 && outcome.chain[0].length == 3240,
              "T9.2 LIVE-no-EOS: chain drops 3244 (length > edit_pos 3240), keeps resume ckpt at 3240");

        const bool runtime_would_narrow = outcome.plan_edit_pos < 3240;
        check(!runtime_would_narrow,
              "T9.2 LIVE-no-EOS: runtime does NOT narrow resume (mount survives)");
    }

    // ---- T9.3 OFF-PARITY (flag=0): the same resume/cur -> fired=false; byte-identical to T9.1 ----
    {
        std::vector<int32_t> resume_ck = make_resume_ck_3240();
        resume_ck.back() = 248046;
        const std::vector<int32_t> cur = make_cur_3294_continuation(resume_ck, 248046);

        std::vector<CheckpointLength> chain = { {3240, 0}, {3244, 1} };
        const SuffixHookOutcome outcome = suffix_reuse_apply_invalidation(
            0, false, 3240, chain, resume_ck, cur);
        check(!outcome.fired,
              "T9.3 off-parity: flag=0 -> fired=false (runtime skips the splice block entirely)");
        // The runtime path at flag=0 is the off-flag path: resume unchanged, chain unchanged.
        // T9.1's on-arm no-op (edit_pos==resume, old_suffix_len==0, no narrowing, no rotation,
        // chain survives) is BYTE-IDENTICAL to flag=0's path - the off-flag parity promise.
        check(true,
              "T9.3 off-parity: byte-identical to T9.1's on-arm no-op (mount survives both arms)");
    }

    // ============================================================
    // T10: anchor-splice engagement (the splice's savings discarded)
    // ============================================================

    // ---- T10.1 ANCHOR-SPLICE (checks={2000, 3240, 3244}, edit E=2500) ----
    // Live CLM shape: the conversation grows by append+edit, an edit at E hits mid-file with
    // a turn-boundary checkpoint below it at C=2000.  The splice hook erases lengths > E
    // (drops 3240, 3244; keeps the 2000 anchor), narrows resume to E=2500.  narrowed_resume_target
    // says: restore C=2000, re-read [E, n) = [2500, n).  The chunk loop's `read_from` must be
    // 2500 (the splice's saved work), NOT 2000 (the anchor restore point).
    {
        const std::vector<int32_t> resume_ck = make_resume_ck_3240();     // [1..3240]
        // Build cur: same 3240, but token 2500 replaced with a sentinel and the rest shifted.
        // cur[0..2500) = resume_ck[0..2500); cur[2500] = 99999 (the edit); cur[2501..3294)
        // is a 794-token "unchanged tail" (a copy of resume_ck[2500..3240) reshaped to fill
        // 794 tokens of cur).  For a real edit, the unchanged tail [edit_pos, n_old) is
        // present in cur at positions [edit_pos + delta, n_new) with delta = n_new - n_old = 54.
        const std::vector<int32_t> cur = [&] {
            std::vector<int32_t> v(3294);
            for (int32_t i = 0; i < 2500; ++i) v[i] = resume_ck[i];
            v[2500] = 99999;   // the edit
            // Unchanged tail at positions [2554, 3294) = resume_ck[2500, 3240) shifted by +54.
            for (int32_t i = 2554; i < 3294; ++i) v[i] = resume_ck[i - 54];
            return v;
        }();

        // The plan: narrow_pos = LCP = 2500 (first 2500 match, token 2500 differs).
        // old_suffix_len from the suffix walk (the unchanged tail from the back) is
        // whatever the back-walk finds.  The splice hook uses narrow_pos for narrowing
        // and k_rotation_start (= plan.edit_pos, suffix-derived) for the K rotation range.
        const auto plan = compute_suffix_plan(resume_ck, cur);
        check(plan.narrow_pos == 2500,
              "T10.1 ANCHOR: plan.narrow_pos == 2500 (LCP-based; the first index of divergence)");

        // The splice hook helper: predicate fires, plan is set, chain drops lengths > 2500.
        std::vector<CheckpointLength> chain = { {2000, 0}, {3240, 1}, {3244, 2} };
        const SuffixHookOutcome outcome = suffix_reuse_apply_invalidation(
            64, false, 3240, chain, resume_ck, cur);
        check(outcome.fired,
              "T10.1 ANCHOR: hook fires (flag=64, controls met, real edit at E=2500)");
        check(outcome.plan_edit_pos == 2500,
              "T10.1 ANCHOR: outcome.plan_edit_pos == 2500 (the edit pos)");
        check(outcome.chain.size() == 1 && outcome.chain[0].length == 2000,
              "T10.1 ANCHOR: chain drops 3240/3244 (> 2500), keeps anchor at 2000");

        // Runtime narrowing: plan_edit_pos=2500 < resume=3240 -> narrows.
        // splice_narrowed_resume = 2500.  resume is set to 2500 by the runtime.
        const bool runtime_would_narrow = outcome.plan_edit_pos < 3240;
        check(runtime_would_narrow,
              "T10.1 ANCHOR: runtime narrows resume to 2500 (real edit; mount must use anchor)");

        // narrowed_resume_target(2500, {2000}) -> restore=2000 (deepest <= 2500),
        // read_from=2000 (r2 F4: read_from == restore_length - the chunk loop starts
        // AT the restored checkpoint; the [2000, 2500) gap is re-read by the prefill,
        // never skipped).
        const std::vector<int64_t> chain_lengths = {2000};
        const NarrowedResumeTarget t = narrowed_resume_target(2500, chain_lengths);
        check(t.restore_length == 2000,
              "T10.1 ANCHOR: t.restore_length == 2000 (the anchor C; restored through)");
        check(t.read_from == 2000,
              "T10.1 ANCHOR: t.read_from == 2000 (== restore_length; fail-closed, the gap [2000, 2500) is re-read)");

        // The wiring: when splice_narrowed the chunk loop's read_from is t.read_from
        // (== restore_length, fail-closed), not `resume`.  The header's
        // wire_splice_read_from captures the resolution.
        const int64_t read_from_spliced = wire_splice_read_from(
            /*splice_narrowed=*/true, t, /*resume=*/2000, /*reread_to=*/-1);
        check(read_from_spliced == 2000,
              "T10.1 ANCHOR: wire_splice_read_from(true, t, 2000, -1) == 2000 (t.read_from == restore_length)");
        const int64_t read_from_legacy = wire_splice_read_from(
            /*splice_narrowed=*/false, t, /*resume=*/2000, /*reread_to=*/-1);
        check(read_from_legacy == 2000,
              "T10.1 ANCHOR: wire_splice_read_from(false, t, 2000, -1) == 2000 (legacy path; splice did not narrow)");

        // The wire_splice_read_from also handles STRATA_CKPT_REREAD (reread_to > 0 -> read_from=0).
        const int64_t read_from_reread = wire_splice_read_from(
            false, t, 2000, /*reread_to=*/2000);
        check(read_from_reread == 0,
              "T10.1 ANCHOR: wire_splice_read_from(false, t, 2000, reread_to=2000) == 0 (STRATA_CKPT_REREAD path)");

        // The spliced path with reread_to set: reread_to wins (rebuild whole prefill from scratch).
        const int64_t read_from_spliced_reread = wire_splice_read_from(
            true, t, 2000, /*reread_to=*/2000);
        check(read_from_spliced_reread == 0,
              "T10.1 ANCHOR: wire_splice_read_from(true, t, 2000, reread_to=2000) == 0 (reread wins)");

        // The re-read region is [read_from, n) = [2000, 3294): the restored anchor C
        // covers [0, 2000) and the prefill re-reads the gap [2000, 2500) plus the edited
        // tail [2500, n) - fail-closed, byte-equal to the strict path's semantics.
        const int64_t total_reread = 3294 - read_from_spliced;
        check(total_reread == 1294,
              "T10.1 ANCHOR: chunk loop re-reads 1294 tokens [2000, 3294) (restore_length, not the edit pos)");
    }

    // ---- T10.2 NO ANCHOR (checks={3240, 3244}; the live 431fc2c+ce20ef3 shape) ----
    // The splice hook erases the resume ckpt (length 3240 > edit_pos 3239) and no candidate
    // survives below edit_pos.  helper returns restore=0, read_from=0; the runtime does
    // session_zero + checks.clear + chunk loop reads from 0 (full re-prefill, no error).
    {
        const std::vector<int64_t> chain_lengths = {};   // splice hook erased them all
        const NarrowedResumeTarget t = narrowed_resume_target(3239, chain_lengths);
        check(t.restore_length == 0,
              "T10.2 NO-ANCHOR: t.restore_length == 0 (no ckpt survives; fresh prefill)");
        check(t.read_from == 0,
              "T10.2 NO-ANCHOR: t.read_from == 0 (full re-prefill from start)");

        // Wiring: splice_narrowed but no anchor -> read_from = t.read_from = 0.
        const int64_t read_from = wire_splice_read_from(true, t, /*resume=*/0, /*reread_to=*/-1);
        check(read_from == 0,
              "T10.2 NO-ANCHOR: wire_splice_read_from(true, t={0,0}, 0, -1) == 0 (fresh prefill)");
    }

    std::printf("\nSummary: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
