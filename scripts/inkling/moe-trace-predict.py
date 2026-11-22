#!/usr/bin/env python3
# Idea 6 analysis: replay a LLAMA_MOE_CACHE_TRACE with dedup keys (Inkling 2: slots p, p+18, p+30 share layer p),
# classify each decode miss by where it was last used, optionally simulate background uploads of CPU-computed
# loop experts (mode "fill"). Usage: moe-trace-predict.py TRACE CACHE_MIB CPU_SHARE base|fill
import sys, importlib.util, collections
spec = importlib.util.spec_from_file_location("mct", "/home/eiselems/Work/llama-cpp/llama.cpp-inkling-perf/scripts/moe-cache-tune.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
n_layer, n_used, groups, recs = m.read_trace(sys.argv[1])
SIZE, SHARE, SKIP = float(sys.argv[2]), float(sys.argv[3]), 40
MODE = sys.argv[4]

# 1) layout check: overlap of slot a with slot b for the same token (decode records)
tok = []  # list of dict il -> set
cur = None
for il, nu, ids in recs:
    if len(ids) != nu: cur = None; continue
    if il == 0: cur = {}; tok.append(cur)
    if cur is not None: cur[il] = set(ids)
tok = [t for t in tok if len(t) == 48][SKIP:]
def ov(a, b): return sum(len(t[a] & t[b]) for t in tok) / len(tok) / n_used
print(f"decode tokens analysed: {len(tok)}")
print("same-token overlap of slot 6 with:", {b: round(ov(6, b), 2) for b in (7, 17, 23, 24, 25, 36, 37, 42)})
print("same-token overlap, pass1->pass2 (p, p+18) avg %.2f, pass2->pass3 (p+18, p+30) %.2f, pass1->pass3 %.2f"
      % (sum(ov(p, p+18) for p in range(6, 18))/12, sum(ov(p+18, p+30) for p in range(6, 18))/12, sum(ov(p, p+30) for p in range(6, 18))/12))
print("previous-token overlap, same slot: avg %.2f" % (sum(len(tok[i][a] & tok[i-1][a]) for i in range(1, len(tok)) for a in range(48)) / (len(tok)-1) / 48 / n_used))

# 2) replay with dedup keys, classify misses
key = {il: il for il in range(48)}
for p in range(6, 18): key[p+18] = p; key[p+30] = p
share = m.f32(SHARE); share_q = int(m.f32(share*65536.0) + 0.5); n_zero = min(n_used, 64)
caches, gof = [], {}
for ig, g in enumerate(groups):
    keys = sorted({key[il] for il in g.layers})
    g2 = m.Group(); g2.__dict__.update(g.__dict__); g2.layers = keys
    ns = m.n_slots_of(g2, SIZE, n_zero, n_used)
    print(f"group {ig}: {len(keys)} key layers, {ns} slots ({100*ns/(len(keys)*g.n_expert):.0f} % of its experts)")
    caches.append(m.ClassLRU(keys, g.n_expert, ns)); gof.update((il, ig) for il in g.layers)
acc = collections.defaultdict(int)
cls = collections.Counter(); tot = collections.Counter()
seen_tok, t_i = -1, -1
hist = {}  # key layer -> list of recent sets (per token)
cur = {}
cpu_done_this_token = collections.defaultdict(set)   # key layer -> experts CPU-computed earlier in this token
for il, nu, ids in recs:
    decode = len(ids) == nu
    if il == 0:
        t_i += 1
        for k, s in cur.items(): hist.setdefault(k, []).append(s)
        cur = {}; cpu_done_this_token.clear()
    c, g, k = caches[gof[il]], groups[gof[il]], key[il]
    slots = c.slot_map[k]
    misses = [e for e in set(ids) if slots[e] >= c.n_slots]
    def n_cpu(n, il=il, c=c):
        v = n*(share_q if c.n_filled == c.n_slots else 0) + acc[il]; acc[il] = v & 0xFFFF; return v >> 16
    before = dict((e, slots[e]) for e in misses)
    c.plan(k, list(ids), nu, n_zero, n_cpu)
    count = decode and t_i > SKIP
    for e in misses:
        to_cpu = slots[e] >= c.n_slots
        if count:
            tot['cpu' if to_cpu else 'upload'] += 1
            prev = hist.get(k, [])
            if e in cpu_done_this_token[k]: cl = 'A same token, earlier pass (CPU-computed there)'
            elif e in cur.get(k, set()): cl = 'A2 same token, earlier pass'
            elif prev and e in prev[-1]: cl = 'B previous token'
            elif any(e in s for s in prev[-8:]): cl = 'C within last 8 tokens'
            elif any(e in s for s in prev[-64:]): cl = 'D within last 64 tokens'
            else: cl = 'E not in last 64 tokens'
            cls[cl] += 1
        if to_cpu: cpu_done_this_token[k].add(e)
    cur.setdefault(k, set()).update(ids)
    # E9 idea: a CPU-computed miss of pass 1/2 of a looped layer is uploaded in the background for the later passes
    if MODE == 'fill' and 6 <= k <= 17 and il < 36:
        late = sorted(e for e in misses if slots[e] >= c.n_slots)
        if late:
            c.plan(k, late, len(late), n_zero, lambda n: 0)
            if count: tot['bg upload'] += len(late)
n = tot['cpu'] + tot['upload']; ntok = len(tok)
lookups = ntok * 48 * n_used
print(f"\n[{MODE}] background uploads per token {tot['bg upload']/ntok:.1f}")
print(f"per decode token: {n/ntok:.1f} misses of {48*n_used} expert uses (hit rate {100*(1-n/lookups):.1f} %), {tot['upload']/ntok:.1f} uploaded, {tot['cpu']/ntok:.1f} on the CPU")
for cl in sorted(cls): print(f"  {cl:52s} {cls[cl]/ntok:5.1f} per token  {100*cls[cl]/n:5.1f} %")
