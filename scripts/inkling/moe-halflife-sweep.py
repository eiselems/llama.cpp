#!/usr/bin/env python3
# Replays a LLAMA_MOE_CACHE_TRACE through the Python mirror of the MoE cache (scripts/moe-cache-tune.py) with the
# Inkling 2 dedup keys (slots p, p+18, p+30 share layer p) for several CRF half-lives and class caps, and prints the
# decode misses per token (uploaded / computed on the CPU). Small caches are where the policy matters most.
# Usage: moe-halflife-sweep.py TRACE CPU_SHARE CACHE_MIB[,CACHE_MIB...] [HALF_LIFE,... [N_CLASS_CAP,...]]
#   N_CLASS_CAP 0 = the class count the half-life implies (as in src/llama-moe-cache.cpp)
import importlib.util, math, os, sys

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("mct", os.path.join(here, "..", "moe-cache-tune.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

SKIP = 40  # decode tokens to warm the cache up before counting


def n_class_of(hl):
    return int(math.log2(1.0 / (1.0 - 2.0 ** (-1.0 / hl)))) + 1


def replay(trace, size_mib, share, hl, cap):
    n_layer, n_used, groups, recs = trace
    key = {il: il for il in range(n_layer)}
    if n_layer == 48:
        for p in range(6, 18):
            key[p + 18] = p; key[p + 30] = p
    share = m.f32(share); share_q = int(math.floor(m.f32(share * 65536.0) + 0.5)); n_zero = min(n_used, 64) if share > 0 else 0

    class LRU(m.ClassLRU):
        CRF_HALF_LIFE = float(hl)
        N_CLASS = cap if cap > 0 else n_class_of(hl)

    caches, gof = [], {}
    for ig, g in enumerate(groups):
        keys = sorted({key[il] for il in g.layers})
        g2 = m.Group(); g2.__dict__.update(g.__dict__); g2.layers = keys
        ns = m.n_slots_of(g2, size_mib, n_zero, n_used)
        caches.append(LRU(keys, g.n_expert, ns) if ns > 0 else None); gof.update((il, ig) for il in g.layers)
    acc = [0] * n_layer
    first = min(il for il, _, _ in recs)
    dec_tok = 0; up = cpu = uses = 0; up_b = cpu_b = 0
    for il, nu, ids in recs:
        nt = len(ids) // nu
        if nt > m.MAX_BATCH:
            continue  # large batches stream the experts, they do not plan the cache
        if il == first:
            dec_tok += nt
        count = dec_tok > SKIP
        c, g, k = caches[gof[il]], groups[gof[il]], key[il]
        if c is None:
            continue

        def n_cpu(n, il=il, c=c):
            v = n * (share_q if c.n_filled == c.n_slots else 0) + acc[il]; acc[il] = v & 0xFFFF; return v >> 16

        n_fill, n_c = c.plan(k, list(ids), nu, n_zero, n_cpu)
        if count:
            up += n_fill; cpu += n_c; uses += len(set(ids))
            up_b += n_fill * g.expert_bytes; cpu_b += n_c * g.expert_bytes
    n = max(dec_tok - SKIP, 1)
    return dict(tokens=n, miss=(up + cpu) / n, up=up / n, cpu=cpu / n, hit=100 * (1 - (up + cpu) / max(uses, 1)),
                up_mb=up_b / n / 2**20, cpu_mb=cpu_b / n / 2**20)


def main():
    trace = m.read_trace(sys.argv[1])
    share = float(sys.argv[2])
    sizes = [float(x) for x in sys.argv[3].split(",")]
    hls = [float(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "32,64,128,256,512,1024").split(",")]
    caps = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "0").split(",")]
    print(f"{'cache MiB':>9} {'half-life':>9} {'classes':>7} {'hit %':>6} {'miss/tok':>8} {'up/tok':>7} {'cpu/tok':>7} {'up MB':>6} {'cpu MB':>6}")
    for size in sizes:
        for hl in hls:
            for cap in caps:
                r = replay(trace, size, share, hl, cap)
                ncls = cap if cap > 0 else n_class_of(hl)
                print(f"{size:9.0f} {hl:9.0f} {ncls:7d} {r['hit']:6.2f} {r['miss']:8.2f} {r['up']:7.2f} {r['cpu']:7.2f} {r['up_mb']:6.1f} {r['cpu_mb']:6.1f}", flush=True)
        print(f"  ({r['tokens']} decode tokens counted)")


if __name__ == "__main__":
    main()
