// jevfields_test — the P-clean wire contract, with no GPU, no model and no server.
//
// Every rejection here is a silent-corruption bug avoided: a split whose lengths do not add up would
// read one field's cue at another field's row and answer a neighbouring question while looking correct.
#include <cstdio>
#include <string>
#include <vector>

#include "strata/program/jevfields.hpp"

using strata::program::FieldSplit;
using strata::program::parse_jevfields;

static int g_fail = 0;
static int g_checks = 0;

static void ok(const char* what, bool cond) {
    ++g_checks;
    if (cond) { std::printf("  ok   %s\n", what); return; }
    std::printf("  FAIL %s\n", what);
    ++g_fail;
}

// 100 tokens, prefix of 40, three fields of 20 each.
static bool parse(const char* s, int64_t n, int64_t pin, FieldSplit& fs) {
    std::string why;
    return parse_jevfields(s, n, pin, fs, why);
}

int main() {
    std::printf("jevfields_test\n");
    FieldSplit fs;

    // 1. The happy path: the lengths add up to the prompt past the prefix.
    ok("three fields of 20 over a 40-token prefix", parse("20,20,20", 100, 40, fs));
    ok("  prefix kept", fs.pin == 40);
    ok("  three lengths", fs.lens.size() == 3);
    ok("  starts are cumulative from the prefix", fs.start(0) == 40 && fs.start(1) == 60 && fs.start(2) == 80);
    ok("  total", fs.total() == 60);

    // 2. One field is legal but pointless; it must still parse so the caller cannot silently get N=1.
    ok("one field", parse("60", 100, 40, fs));

    // 3. The cap.
    ok("32 fields", parse("1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1", 72, 40, fs));
    ok("33 fields refused", !parse("1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1", 73, 40, fs));

    // 4. Sums that do not add up: refused WHOLE, never truncated to a prefix of the list.
    ok("one token short refused", !parse("20,20,19", 100, 40, fs));
    ok("  and leaves no partial split", fs.lens.empty());
    ok("one token long refused", !parse("20,20,21", 100, 40, fs));
    ok("  and leaves no partial split", fs.lens.empty());

    // 5. Degenerate splits.
    ok("a zero-length field refused", !parse("20,0,40", 100, 40, fs));
    ok("empty list refused", !parse("", 100, 40, fs));
    ok("null refused", !parse(nullptr, 100, 40, fs));
    ok("no prefix refused", !parse("60", 100, 0, fs));
    ok("prefix == n refused", !parse("", 40, 40, fs));

    // 6. Malformed text: anything that is not a list of numbers.
    ok("letters refused", !parse("20,x,20", 100, 40, fs));
    ok("negative refused", !parse("20,-20,60", 100, 40, fs));
    ok("trailing comma tolerated", parse("20,20,20,", 100, 40, fs));
    ok("spaces refused", !parse("20, 20,20", 100, 40, fs));
    ok("an absurd length refused", !parse("999999999999999999999,1", 100, 40, fs));

    std::printf("jevfields_test: %d checks, %d failed\n", g_checks, g_fail);
    std::printf(g_fail == 0 ? "jevfields_test: all checks passed\n" : "jevfields_test: FAILURES\n");
    return g_fail == 0 ? 0 : 1;
}
