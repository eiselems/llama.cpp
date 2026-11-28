#!/usr/bin/env python3
# Per-layer MoE cache budget: replays a decode LLAMA_MOE_CACHE_TRACE (Inkling 2, 48 layer applications) through the
# Python mirror of the cache (scripts/moe-cache-tune.py) with
#  w  = eviction score weight for the experts of layers 0-5 (>1: evicted earlier, <1: kept longer)
#  se = CPU share of the misses of layers 0-5 (1.0 = never uploaded, i.e. out of the cache)
#  so = CPU share of the other layers
# Prints misses per token and two costs: sum = uploads + CPU experts serial, max = per layer max(upload, CPU), which
# matches the server (the uploads of a layer run while its CPU experts compute; 2026-11-28 A/B).
# Usage: moe-budget-sim.py TRACE CACHE_MIB "w,se,so;w,se,so;..."
import importlib.util, math, os, sys
from multiprocessing import Pool

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("mct", os.path.join(here, "..", "moe-cache-tune.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

SKIP = 40
EARLY = set(range(6))
PCIE, RAM = 26.3e9, 48.1e9  # measured 2026-11-27


def replay(args):
    path, size_mib, w, se, so = args
    n_layer, n_used, groups, recs = m.read_trace(path)
    key = {il: il for il in range(n_layer)}
    for p in range(6, 18):
        key[p + 18] = p; key[p + 30] = p
    n_zero = n_used

    def q(s):
        return int(math.floor(m.f32(m.f32(s) * 65536.0) + 0.5))
    share_q = {il: q(se if il in EARLY else so) for il in range(n_layer)}

    class LRU(m.ClassLRU):
        CRF_HALF_LIFE = 64.0
        N_CLASS = int(math.log2(1.0 / (1.0 - 2.0 ** (-1.0 / 64.0)))) + 1

        def victim(self, il):
            res, res_score = -1, -1.0
            for c in range(self.N_CLASS):
                s = self.head[c]
                # scan the whole class: with weights the head is not always the best victim of its class
                while s >= 0:
                    k = self.key_of[s]
                    if not self.is_selected(il, k):
                        if k < 0:
                            return s
                        score = math.ldexp(float(self.clock[k // self.n_expert] - self.last_use[k]), -c)
                        if k // self.n_expert in EARLY:
                            score *= w
                        if score > res_score:
                            res, res_score = s, score
                        if w == 1.0:
                            break  # same as the C++ cache: the head of each class only
                    s = self.next[s]
            return res

    caches, gof = [], {}
    for ig, g in enumerate(groups):
        keys = sorted({key[il] for il in g.layers})
        g2 = m.Group(); g2.__dict__.update(g.__dict__); g2.layers = keys
        ns = m.n_slots_of(g2, size_mib, n_zero, n_used)
        caches.append(LRU(keys, g.n_expert, ns) if ns > 0 else None); gof.update((il, ig) for il in g.layers)
    acc = [0] * n_layer
    first = min(il for il, _, _ in recs)
    dec = 0
    per = {}  # group name -> [uses, up, cpu]
    up_b = cpu_b = 0
    mx = 0.0  # sum over layer applications of max(upload, CPU): the two run at the same time
    for il, nu, ids in recs:
        nt = len(ids) // nu
        if nt > m.MAX_BATCH:
            continue
        if il == first:
            dec += nt
        c, g = caches[gof[il]], groups[gof[il]]
        if c is None:
            continue

        def n_cpu(n, il=il, c=c):
            v = n * (share_q[il] if c.n_filled == c.n_slots else 0) + acc[il]; acc[il] = v & 0xFFFF; return v >> 16

        n_fill, n_c = c.plan(key[il], list(ids), nu, n_zero, n_cpu)
        if dec > SKIP:
            name = '0-5' if il < 6 else '18-23' if 18 <= il < 24 else 'loop'
            p = per.setdefault(name, [0, 0, 0])
            p[0] += len(set(ids)); p[1] += n_fill; p[2] += n_c
            up_b += n_fill * g.expert_bytes; cpu_b += n_c * g.expert_bytes
            mx += max(n_fill * g.expert_bytes / PCIE, n_c * g.expert_bytes / RAM)
    n = dec - SKIP
    uses = sum(p[0] for p in per.values()); up = sum(p[1] for p in per.values()); cpu = sum(p[2] for p in per.values())
    return dict(w=w, se=se, so=so, n=n, hit=100 * (1 - (up + cpu) / uses), up=up / n, cpu=cpu / n,
                up_ms=up_b / n / PCIE * 1e3, mx_ms=mx / n * 1e3, cpu_ms=cpu_b / n / RAM * 1e3,
                per={k: (100 * (1 - (v[1] + v[2]) / v[0]), v[1] / n, v[2] / n) for k, v in per.items()})


if __name__ == "__main__":
    path, size = sys.argv[1], float(sys.argv[2])
    cfgs = [tuple(float(x) for x in c.split(',')) for c in sys.argv[3].split(';')]
    with Pool(min(len(cfgs), 10)) as pool:
        res = pool.map(replay, [(path, size, *c) for c in cfgs])
    print(f"{'w':>5} {'se':>5} {'so':>5} {'hit%':>6} {'up/t':>6} {'cpu/t':>6} {'upms':>5} {'cpums':>5} {'sum':>5} {'max':>5} | per group hit% up cpu")
    for r in res:
        per = '  '.join(f"{k} {v[0]:.1f} {v[1]:.1f} {v[2]:.1f}" for k, v in sorted(r['per'].items()))
        print(f"{r['w']:5.2f} {r['se']:5.2f} {r['so']:5.2f} {r['hit']:6.2f} {r['up']:6.2f} {r['cpu']:6.2f} {r['up_ms']:5.2f} {r['cpu_ms']:5.2f} {r['up_ms'] + r['cpu_ms']:5.2f} {r['mx_ms']:5.2f} | {per}")
    print(f"({res[0]['n']} decode tokens counted, cache {size:.0f} MiB)")
