// jevpipe_test — the P gate, with no GPU, no model and no server.
//
// The bug this locks down: `jevpos=` (arm B) and `--pipeline-windows` used to be mutually exclusive, so
// every one-prompt/N-answers request silently took the serial window loop and paid ~2.6x per prompt token.
// The fix is opt-in, which means the interesting cases are the REFUSALS: an env var that is not exactly "1"
// must leave every other request on today's byte-identical path.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "strata/program/jevpipe.hpp"

using strata::program::PipeReadAsk;
using strata::program::pipe_read_allowed;
using strata::program::pipe_capture_enabled;

static int g_fail = 0;
static int g_checks = 0;

static void check(const char* what, bool got, bool want) {
    ++g_checks;
    if (got == want) { std::printf("  ok   %s\n", what); return; }
    std::printf("  FAIL %s: got %s want %s\n", what, got ? "true" : "false", want ? "true" : "false");
    ++g_fail;
}

// A pipelined 2-stage run, no rows named: the plain throughput path.
static PipeReadAsk plain() { return PipeReadAsk{true, 2, 1, false, false, false}; }
// The same run with rows named and the new flag on.
static PipeReadAsk rows_cap() { return PipeReadAsk{true, 2, 1, true, false, true}; }

int main() {
    std::printf("jevpipe_test\n");

    // 1. Nothing named: the gate answers exactly as before, flag or no flag.
    check("plain request, flag off", pipe_read_allowed(plain()), true);
    { PipeReadAsk a = plain(); a.pipe_cap = true; check("plain request, flag on", pipe_read_allowed(a), true); }

    // 2. The regression: rows named, flag off -> serial.  This is today's behaviour and must not move.
    { PipeReadAsk a = rows_cap(); a.pipe_cap = false; check("rows named, flag off -> serial", pipe_read_allowed(a), false); }

    // 3. The feature: rows named, flag on -> pipelined read, rows captured in it.
    check("rows named, flag on -> pipelined", pipe_read_allowed(rows_cap()), true);

    // 4. STRATA_LOGPOS still wins: it reads every window's logits and is serial by contract.
    { PipeReadAsk a = rows_cap(); a.logpos = true; check("LOGPOS refuses the pipeline", pipe_read_allowed(a), false); }
    { PipeReadAsk a = plain(); a.logpos = true; check("LOGPOS refuses even with no rows", pipe_read_allowed(a), false); }

    // 5. Only the 2-stage pipelined shape; 3+ stages and depth 0 stay serial.
    { PipeReadAsk a = rows_cap(); a.n_stages = 3; check("3 stages -> serial", pipe_read_allowed(a), false); }
    { PipeReadAsk a = rows_cap(); a.n_stages = 1; check("1 stage -> serial", pipe_read_allowed(a), false); }
    { PipeReadAsk a = rows_cap(); a.pl_pw = 0;    check("depth 0 -> serial", pipe_read_allowed(a), false); }
    { PipeReadAsk a = rows_cap(); a.pipe = false; check("--pipeline-windows off -> serial", pipe_read_allowed(a), false); }

    // 6. The env read is exact: "1" only.  Anything else leaves today's path in place.
    struct EnvCase { const char* v; bool want; };
    const EnvCase cases[] = {{"1", true}, {"0", false}, {"", false}, {"true", false}, {"yes", false},
                             {"11", false}, {" 1", false}, {"1 ", false}};
    for (const EnvCase& c : cases) {
        setenv("STRATA_JEV_PIPE_CAP", c.v, 1);
        const std::string name = std::string("env \"") + c.v + "\" -> " + (c.want ? "on" : "off");
        check(name.c_str(), pipe_capture_enabled(), c.want);
    }
    unsetenv("STRATA_JEV_PIPE_CAP");
    check("env unset -> off", pipe_capture_enabled(), false);

    std::printf("jevpipe_test: %d checks, %d failed\n", g_checks, g_fail);
    std::printf(g_fail == 0 ? "jevpipe_test: all checks passed\n" : "jevpipe_test: FAILURES\n");
    return g_fail == 0 ? 0 : 1;
}
