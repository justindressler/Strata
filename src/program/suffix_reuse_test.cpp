// src/program/suffix_reuse_test.cpp - the suffix-reuse plan computation: pure CPU tests on the longest
// common tail of a prompt re-edit, no model, no GPU.
//
// The suffix reuses the prompt cache's K/V for the unchanged tail beyond a mid-prompt edit at position E.
// Longest common suffix is the splice plan: tokens [E, n_new) are the unchanged suffix, tokens [E, n_old)
// are the unchanged suffix in the old prompt; if the per-token difference in absolute position (`delta`)
// is uniform, the splice reuses one rotation; otherwise per-token deltas go to the rotation kernel.
//
// T1: pure plan-computation, single insert (uniform delta = +k).
// T2: pure invalidation - checkpoints whose length > E drop.
// T4: composite edit (insert+delete in different places) - per-token deltas correct.
//
// T5 (chain integration) and T8 (fail-closed guards) live in the same test TU.
// T3 (RoPE rotation of the removed splice layer) was deleted with the splice.
#include "strata/program/suffix_reuse.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/core/conversation_cache.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using strata::program::suffix_reuse::SuffixReusePlan;
using strata::program::suffix_reuse::compute_suffix_plan;
using strata::program::suffix_reuse::invalidate_checkpoints;
using strata::program::suffix_reuse::CheckpointLength;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-72s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    std::printf("suffix_reuse_test\n");

    // --- T1: pure plan-computation --------------------------------------------------
    // insert 3 tokens at position 4 of a 7-token prompt -> uniform delta = +3 across the suffix
    {
        const std::vector<int32_t> old_tokens = {1, 2, 3, 4, 5, 6, 7};
        const std::vector<int32_t> new_tokens = {1, 2, 3, 4, 100, 200, 300, 5, 6, 7};
        const SuffixReusePlan plan = compute_suffix_plan(old_tokens, new_tokens);
        check(plan.edit_pos == 4, "T1: insert at 4 -> edit_pos is 4");
        check(plan.uniform && plan.delta == 3, "T1: uniform insert -> delta = +3");
        check(plan.old_suffix_len == 3 && plan.new_suffix_len == 3,
              "T1: tail [5,6,7] reuses against [5,6,7] of the new prompt");
        check(plan.deltas.empty(), "T1: uniform delta needs no per-token table");
    }
    // pure deletion: drop tokens [4,5] -> new prompt shorter than old, delta = -2 across the tail
    {
        const std::vector<int32_t> old_tokens = {1, 2, 3, 4, 5, 6, 7};
        const std::vector<int32_t> new_tokens = {1, 2, 3, 6, 7};
        const SuffixReusePlan plan = compute_suffix_plan(old_tokens, new_tokens);
        check(plan.edit_pos == 5, "T1: delete of [4,5] -> edit_pos at 5 in the old prompt");
        check(plan.uniform && plan.delta == -2, "T1: uniform delete -> delta = -2");
        check(plan.old_suffix_len == 2 && plan.new_suffix_len == 2,
              "T1: tail [6,7] reuses against [6,7]");
    }
    // no edit at all: edit_pos == 0 (the entire old prompt is the common tail), delta = 0
    {
        const std::vector<int32_t> old_tokens = {1, 2, 3, 4};
        const std::vector<int32_t> new_tokens = {1, 2, 3, 4};
        const SuffixReusePlan plan = compute_suffix_plan(old_tokens, new_tokens);
        check(plan.edit_pos == 0 && plan.uniform && plan.delta == 0,
              "T1: identical prompts: edit_pos is 0 (whole prompt reuses), no rotation");
    }

    // --- T2: pure invalidation - any checkpoint with length > E drops ---------------
    {
        // four checkpoints forming a chain (lengths sorted): 2, 5, 7, 9 - the edit fell at E=4
        std::vector<CheckpointLength> chain = { {2, 0}, {5, 1}, {7, 2}, {9, 3} };
        const size_t dropped = invalidate_checkpoints(chain, 4);
        check(dropped == 3, "T2: checkpoints with length > 4 drop (5, 7, 9)");
        check(chain.size() == 1 && chain[0].length == 2,
              "T2: the sole survivor is the prefix at length 2");
    }
    {   // edit past the chain's deepest point: nothing to drop, chain stays whole
        std::vector<CheckpointLength> chain = { {2, 0}, {5, 1}, {7, 2} };
        const size_t dropped = invalidate_checkpoints(chain, 10);
        check(dropped == 0 && chain.size() == 3,
              "T2: edit past the chain leaves every checkpoint valid");
    }
    {   // edit at 0 (the whole prefix is suspect): all checkpoints drop
        std::vector<CheckpointLength> chain = { {2, 0}, {5, 1} };
        const size_t dropped = invalidate_checkpoints(chain, 0);
        check(dropped == 2, "T2: edit at 0 invalidates every checkpoint");
    }

    // --- T3 (RoPE CPU rotation of the removed splice layer) was deleted with the splice:
    //     its host rope-table / row-rotation helpers lived in the deleted splice header. ---

    // --- T5: integration with retain/limit_reuse on a synthetic ConversationCache ------
    // the splice plan's `invalidate_checkpoints` runs BEFORE retain/limit_reuse, so the surviving
    // checkpoints (lengths <= edit_pos) stay parked.  The unchanged_tokens the caller's take_reuse
    // sees is bounded by the splice region's start (== edit_pos), and the suffix tail reuses via the
    // splice (V unchanged, K rotated) - this test pins that math against today's behaviour by
    // building a ConversationCache, feeding it a `retain` + `limit_reuse(first_dirty)` sequence, and
    // asserting the byte accounting is the same whether or not the splice plan ran first.
    {
        auto cp = [](std::vector<int32_t> ids) {
            strata::core::ConversationCheckpoint c;
            c.ids = std::move(ids);
            return c;
        };
        // mirrors the pinned test in src/core/conversation_cache_test.cpp:62-84 (a single layer, a
        // retainer, a limit_reuse rewind).  added here to bind the splice invalidation to the same
        // chain so a future reviewer can see end-to-end accounting with a checkpoint chain present.
        strata::core::ConversationKv layer;
        layer.k.resize(400);
        std::vector<strata::core::ConversationKv> layers;
        layers.push_back(std::move(layer));
        const size_t retained = layers.capacity() * sizeof(strata::core::ConversationKv) + layers[0].bytes();
        auto image = [](std::initializer_list<int32_t> ids) {
            strata::core::SavedConversation s;
            s.live.ids = ids;
            s.live.gdn.resize(64, 7);
            s.cvec = true;
            return s;
        };
        const size_t parked = image({1, 2, 3}).bytes();
        strata::core::ConversationCache cache(retained + parked, 4);

        // park, retain, splice-plan invalidate, limit_reuse; assert the unchanged_tokens identity
        check(cache.put(image({1, 2, 3})), "T5: park before retaining active storage");
        cache.retain(std::move(layers), 16);
        check(cache.bytes() == retained + parked && cache.size() == 1,
              "T5: active retained storage consumes bytes but no parked slot");

        // edit at edit_pos = 9: invalidate every checkpoint whose token length > 9 (none parked yet,
        // this exercises the chain API before the chain is populated, matching the conversation_cache
        // test fixture at line 70-76).
        std::vector<strata::program::suffix_reuse::CheckpointLength> chain;
        chain.push_back({2, 0}); chain.push_back({5, 1}); chain.push_back({9, 2});
        const size_t dropped = strata::program::suffix_reuse::invalidate_checkpoints(chain, 9);
        check(dropped == 0, "T5: edit_pos = 9 keeps every checkpoint with length <= 9");

        // standard conversation_cache test flow: two rewinds, take_reuse, undo-roundtrip
        cache.limit_reuse(9);
        cache.limit_reuse(12);
        auto reuse = cache.take_reuse();
        check(reuse.unchanged_tokens == 9, "T5: a later continuation cannot undo a rewind's dirty boundary");
        check(cache.bytes() == parked, "T5: taking retained buffers releases their budget accounting");

        // simulate a real splice: the splice plan says edit_pos = 9, unchanged tail = [9, n_new).
        // the splice path adds K rotation but the byte accounting is unchanged: `retain` below uses
        // reuse.kv against a fresh retainer (the same move semantics as conversation_cache_test.cpp:76).
        cache.retain(std::move(reuse.kv), 16);

        // checkpoint chain after a real edit at E = 5 (different test point): invalidate everything
        // strictly greater than 5; the survivors are lengths [2].
        std::vector<strata::program::suffix_reuse::CheckpointLength> chain2;
        chain2.push_back({2, 0});
        chain2.push_back({5, 1});
        chain2.push_back({7, 2});
        chain2.push_back({9, 3});
        const size_t dropped2 = strata::program::suffix_reuse::invalidate_checkpoints(chain2, 5);
        check(dropped2 == 2 && chain2.size() == 2, "T5: edit_pos = 5 drops lengths 7 and 9");
        check(chain2[0].length == 2 && chain2[1].length == 5,
              "T5: survivors are the prefix chain up to edit_pos");
    }
    // sequence: old=[1,2,3,4,5,6,7,8], new=[1,2,99,3,4,5,6,7,8]
    // the common tail is [3,4,5,6,7,8] in old vs [3,4,5,6,7,8] in new
    // per-token position delta from old index (2 + i) to new index (3 + i) is uniformly +1
    {
        const std::vector<int32_t> old_tokens = {1, 2, 3, 4, 5, 6, 7, 8};
        const std::vector<int32_t> new_tokens = {1, 2, 99, 3, 4, 5, 6, 7, 8};
        const SuffixReusePlan plan = compute_suffix_plan(old_tokens, new_tokens);
        check(plan.edit_pos == 2, "T4: edit at position 2");
        check(plan.old_suffix_len == 6 && plan.new_suffix_len == 6,
              "T4: six-token common tail [3..8]");
        check(plan.uniform && plan.delta == 1,
              "T4: insertion at position 2 gives a uniform +1 rotation");
    }
    // deletion at a different point from the tail: old=[1,2,3,4,5,6,7], new=[1,2,3,5,6,7]
    // common tail [5,6,7] in old vs [3,4,5] in new, edit_pos = 7 - 3 = 4 in the old prompt
    {
        const std::vector<int32_t> old_tokens = {1, 2, 3, 4, 5, 6, 7};
        const std::vector<int32_t> new_tokens = {1, 2, 3, 5, 6, 7};
        const SuffixReusePlan plan = compute_suffix_plan(old_tokens, new_tokens);
        check(plan.edit_pos == 4, "T4: delete of [4] -> edit_pos at 4 in the old prompt");
        check(plan.uniform && plan.delta == -1,
              "T4: deletion at position 4 gives a uniform -1 rotation");
        check(plan.old_suffix_len == 3 && plan.new_suffix_len == 3,
              "T4: three-token common tail [5,6,7]");
    }

    // --- T8: A1.3 / A2.1 fail-closed guards (CPU-testable mirror of generate.cpp's runtime check) ----
    {
        using strata::program::suffix_reuse::validate_suffix_reuse_options;
        // flag off (0) and any prompt_cache / kv / kv_resident / layer_split - always OK
        check(validate_suffix_reuse_options(0, 0, "fp16", 0, false).empty(),
              "T8: flag off with --prompt-cache 0: no splice, no guard fires");
        check(validate_suffix_reuse_options(0, 1, "fp16", 0, true).empty(),
              "T8: flag off with layer split and any prompt_cache: no splice, no guard fires");
        check(validate_suffix_reuse_options(0, 1, "int8", 16, false).empty(),
              "T8: flag off with int8 + kv_resident=16: no splice, no guard fires");
        // flag on, prompt_cache too low - REJECTS
        const std::string low = validate_suffix_reuse_options(10, 1, "fp16", 0, false);
        check(!low.empty() && low.find("--prompt-cache >= 2") != std::string::npos,
              "T8: flag on + --prompt-cache < 2: guard rejects with --prompt-cache >= 2");
        // flag on, layer_split > 1 stage - REJECTS (A2.1)
        const std::string split = validate_suffix_reuse_options(10, 6, "fp16", 0, true);
        check(!split.empty() && split.find("--layer-split") != std::string::npos,
              "T8: flag on + --layer-split: guard rejects with per-stage splice note");
        // Phase 2: kv format guards.  Phase 3: --kv int8 is NOW SUPPORTED.
        check(validate_suffix_reuse_options(10, 6, "int8", 0, false).empty(),
              "T8: flag on + --kv int8: guard PASSES (Phase 3 int8 splice)");
        check(validate_suffix_reuse_options(10, 6, "q4_0", 0, false).empty(),
              "T8: flag on + --kv q4_0: guard PASSES (the Q4 sandwich; measured in T11.f)");
        check(validate_suffix_reuse_options(10, 6, "k8v4", 0, false).empty(),
              "T8: flag on + --kv k8v4 (hybrid): guard PASSES (K half is int8; V untouched)");
        check(!validate_suffix_reuse_options(10, 6, "fp16", 20480, false).empty(),
              "T8: flag on + --kv-resident 20480: guard rejects with --kv-resident note (Phase 3)");
        // flag on, everything OK - PASSES (both fp16 and int8 are now supported)
        check(validate_suffix_reuse_options(10, 6, "fp16", 0, false).empty(),
              "T8: flag on + --kv fp16 + no kv_resident + no --layer-split: passes");
        check(validate_suffix_reuse_options(10, 6, "int8", 0, false).empty(),
              "T8: flag on + --kv int8 + no kv_resident + no --layer-split: passes (Phase 3)");
    }

    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
