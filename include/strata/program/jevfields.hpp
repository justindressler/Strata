#pragma once
/// Arm P-clean: one request, N independent field suffixes over one shared prefix.
///
/// The shape (and why it exists): a multi-field decision request today costs ~200 ms PER FIELD of
/// admission - measured 2026-10-23 on the 3090, a 66-token request and an 85-token request both take
/// ~205 ms, so the 19 extra tokens are free and the per-request fixed cost is not.  N fields therefore
/// cost N admissions even though each field's own suffix is ~35 tokens.  P-clean sends the prefix once
/// and all N suffixes in ONE request; the engine reads the prefix, then for each field rewinds to the
/// prefix and reads only that field's suffix, so the fields stay independent (no field sees another's
/// cue - which is what arm B's single linear prompt cannot promise) and only one admission is paid.
///
/// The wire: `pin=P` (the shared prefix, already exists) plus `jevfields=l1,l2,...`, with the ids array
/// holding the prefix followed by the N suffixes back to back.  The contract is validated HERE, before
/// the request flow sees it, because every failure mode of a wrong split is silent: a length that does
/// not add up would read one field's cue at another field's row and answer a neighbour's question while
/// looking correct.  Anything doubtful leaves the list EMPTY, which means "not a P-clean request" and
/// the server serves the field the ordinary way.
#include <cstdint>
#include <string>
#include <vector>

namespace strata::program {

/// The engine's cap on named rows, reused as the cap on fields: a longer list than the row readout can
/// address would be useless even if it were parsed.
inline constexpr int kMaxJevFields = 32;

struct FieldSplit {
    int64_t pin = 0;                    ///< tokens of the shared prefix (== `pin=`)
    std::vector<int64_t> lens;          ///< each field's suffix length, in order
    int64_t total() const { int64_t t = 0; for (int64_t v : lens) t += v; return t; }
    int64_t start(int i) const { int64_t s = pin; for (int k = 0; k < i; ++k) s += lens[(size_t) k]; return s; }
};

/// Parse and validate `jevfields=`.  Returns false and leaves `out` empty on ANY doubt.
///
/// Rules, all fail-closed:
///   * `pin` must be >= 1: without a shared prefix there is nothing to read once, so this is not the
///     shape (a caller that wants N prompts wants N requests).
///   * every field length must be >= 1.
///   * at most kMaxJevFields fields.
///   * the lengths must sum EXACTLY to `n - pin`: one token short and the last field's cue is never run,
///     one token long and the read runs off the end of the ids array.
inline bool parse_jevfields(const char* s, int64_t n, int64_t pin, FieldSplit& out, std::string& why) {
    out = FieldSplit{};
    out.pin = pin;
    if (s == nullptr || *s == '\0') { why = "empty"; return false; }
    if (pin < 1) { why = "no shared prefix (pin must be >= 1)"; return false; }
    if (pin >= n) { why = "the prefix is the whole prompt"; return false; }

    std::vector<int64_t> lens;
    const char* p = s;
    while (*p != '\0') {
        if (*p == ',') { p++; continue; }
        if (*p < '0' || *p > '9') { why = "not a list of lengths"; out.lens.clear(); return false; }
        int64_t v = 0;
        int digits = 0;
        while (*p >= '0' && *p <= '9') {
            if (++digits > 18) { why = "a length that cannot be a token count"; out.lens.clear(); return false; }
            v = v * 10 + (*p - '0');
            p++;
        }
        if (v < 1) { why = "a field with no tokens"; out.lens.clear(); return false; }
        lens.push_back(v);
        if ((int) lens.size() > kMaxJevFields) { why = "more fields than the row readout can address"; out.lens.clear(); return false; }
    }
    if (lens.empty()) { why = "no fields"; return false; }

    int64_t sum = 0;
    for (int64_t v : lens) sum += v;
    if (sum != n - pin) {
        why = "the lengths do not add up to the prompt past the prefix";
        out.lens.clear();                    // never a partial split: that reads a neighbour's row
        return false;
    }
    out.lens = std::move(lens);
    why.clear();
    return true;
}

}  // namespace strata::program
