#pragma once
/// Arm P (M3.5): join the pipelined prompt read and the jev row capture.
///
/// Why this exists: the pipelined read is the throughput path and the `jevpos=` row capture is the
/// correctness path, and before P they were mutually exclusive — the gate in generate.cpp sent every
/// `jevpos=` request to the serial window loop, so a one-prompt/N-answers request paid ~2.6x per prompt
/// token (measured 2026-10-23: 7.6 ms/token for arm B at N=2 against 2.9 for arm C).  P joins them behind
/// STRATA_JEV_PIPE_CAP=1.  The decision lives in one GPU-free function so it can be unit-tested with plain
/// g++ — no GPU, no model, no server — and so the gate and the docs cannot drift apart.
#include <cstdlib>
#include <cstring>

namespace strata::program {

/// Everything the gate needs, passed explicitly so the rule is testable without the process env.
struct PipeReadAsk {
    bool pipe;        ///< --pipeline-windows on and the stage split allows it
    int  n_stages;    ///< only 2 stages is a pipelined shape (3+ reads serially by design)
    int  pl_pw;       ///< the pipeline window depth
    bool want_rows;   ///< the request named rows (`jevpos=` non-empty)
    bool logpos;      ///< STRATA_LOGPOS reads every window's logits: serial by contract
    bool pipe_cap;    ///< STRATA_JEV_PIPE_CAP=1
};

/// True when the pipelined read may run.  Note the two unconditional refusals come first: STRATA_LOGPOS
/// and a non-pipelined shape both keep today's path whatever else is set.
inline bool pipe_read_allowed(const PipeReadAsk& a) {
    if (a.logpos) return false;
    if (!a.pipe || a.n_stages != 2 || a.pl_pw < 1) return false;
    if (!a.want_rows) return true;   // nothing to capture: unchanged behaviour for every other request
    return a.pipe_cap;               // rows + pipeline: opt-in only
}

/// STRATA_JEV_PIPE_CAP=1 exactly.  "0", "", "true" and "yes" are all OFF, so neither an unset variable nor
/// a plausible-looking typo silently leaves the serial path.
inline bool pipe_capture_enabled() {
    const char* e = std::getenv("STRATA_JEV_PIPE_CAP");
    return e != nullptr && e[0] == '1' && e[1] == '\0';
}

}  // namespace strata::program
