#!/usr/bin/env python3
# Renders docs/inkling-perf/progress.svg from docs/inkling-perf/progress.tsv (stdlib only, GitHub renders the SVG inline).
#
# progress.tsv columns: step  commit  change  vram_mib  pp1k  pp5k  tg1k  tg5k  args
# Two panels share the x axis (one row per step): prefill on top, decode below. They have different
# scales, so they get their own y axes instead of one dual-axis chart.
import csv, html, os, sys

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs', 'inkling-perf')
rows = list(csv.DictReader(open(os.path.join(root, 'progress.tsv')), delimiter='\t'))
if not rows:
    sys.exit('no rows')

W, PAD_L, PAD_R = 960, 64, 150
PANEL_H, GAP, TOP = 230, 54, 56
X_LAB = 120                                   # room for the rotated step labels under the lower panel
H = TOP + 2*PANEL_H + GAP + X_LAB
plot_w = W - PAD_L - PAD_R
n = len(rows)
xs = [PAD_L + (plot_w*(i + 0.5)/n) for i in range(n)]

# palette: categorical slots 1-2, light + dark (validated: CVD dE >= 24 in both modes)
CSS = """
:root { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8984; --grid:#e6e5e0; --s1:#2a78d6; --s2:#eb6834; }
@media (prefers-color-scheme: dark) { :root { --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#8f8e88; --grid:#33332f; --s1:#3987e5; --s2:#d95926; } }
text { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; fill: var(--ink2); font-size: 12px; }
.title { fill: var(--ink); font-size: 15px; font-weight: 600; }
.ptitle { fill: var(--ink); font-size: 13px; font-weight: 600; }
.tick { fill: var(--muted); font-size: 11px; }
.grid { stroke: var(--grid); stroke-width: 1; }
.base { stroke: var(--muted); stroke-width: 1; }
.l1 { stroke: var(--s1); fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.l2 { stroke: var(--s2); fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; stroke-dasharray: 6 4; }
.m1 { fill: var(--s1); stroke: var(--surface); stroke-width: 2; }
.m2 { fill: var(--s2); stroke: var(--surface); stroke-width: 2; }
.val { fill: var(--ink); font-size: 11px; font-weight: 600; }
"""

def nice_max(v):
    for step in (1, 2, 2.5, 5):
        for mag in (1, 10, 100, 1000, 10000):
            m = step*mag
            if m >= v*1.08:
                return m
    return v*1.1

def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
       f'aria-label="Inkling-2-Mini prefill and decode speed per optimization step">',
       f'<style>{CSS}</style>', f'<rect width="{W}" height="{H}" fill="var(--surface)"/>',
       f'<text class="title" x="{PAD_L}" y="26">Inkling-2-Mini on RTX 3070 8 GB - llama.cpp fork, per optimization step</text>',
       f'<text x="{PAD_L}" y="44">tokens/s, higher is better. UD-Q4_K_M, 32k ctx, q8_0 KV, experts in RAM. Hover a point for the exact value.</text>']

def panel(y0, title, keys, names, unit_note):
    vals = [f(r[k]) for r in rows for k in keys]
    vmax = nice_max(max(v for v in vals if v is not None))
    y = lambda v: y0 + PANEL_H - (v/vmax)*PANEL_H
    out.append(f'<text class="ptitle" x="{PAD_L}" y="{y0 - 10}">{title}</text>')
    for i in range(5):
        v = vmax*i/4
        cls = 'base' if i == 0 else 'grid'
        out.append(f'<line class="{cls}" x1="{PAD_L}" x2="{PAD_L + plot_w}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        out.append(f'<text class="tick" x="{PAD_L - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v:g}</text>')
    for si, (k, name) in enumerate(zip(keys, names), start=1):
        pts = [(xs[i], y(f(r[k]))) for i, r in enumerate(rows) if f(r[k]) is not None]
        if len(pts) > 1:
            out.append(f'<polyline class="l{si}" points="' + ' '.join(f'{px:.1f},{py:.1f}' for px, py in pts) + '"/>')
        for i, r in enumerate(rows):
            v = f(r[k])
            if v is None:
                continue
            tip = html.escape(f"{r['step']} {r['change']} ({r['commit']}): {name} {v:g} t/s, VRAM {r['vram_mib']} MiB")
            out.append(f'<circle class="m{si}" cx="{xs[i]:.1f}" cy="{y(v):.1f}" r="5"><title>{tip}</title></circle>')
        # direct label at the last point: the series name and its current value
        last = max(i for i, r in enumerate(rows) if f(r[k]) is not None)
        lv = f(rows[last][k])
        dy = -9 if si == 1 else 15
        out.append(f'<text class="val" x="{xs[last] + 10:.1f}" y="{y(lv) + dy/3:.1f}">{name} {lv:g}</text>')
    # first-to-last gain for series 1
    first = next(f(r[keys[0]]) for r in rows if f(r[keys[0]]) is not None)
    lastv = [f(r[keys[0]]) for r in rows if f(r[keys[0]]) is not None][-1]
    out.append(f'<text class="tick" x="{PAD_L + plot_w}" y="{y0 - 10}" text-anchor="end">{names[0]}: {first:g} -> {lastv:g} ({lastv/first:.1f}x){unit_note}</text>')

y_pp = TOP + 30
panel(y_pp, 'Prefill (prompt processing)', ['pp1k', 'pp5k'], ['1k prompt', '4.8k prompt'], '')
y_tg = y_pp + PANEL_H + GAP
panel(y_tg, 'Decode (token generation, 128 tokens)', ['tg1k', 'tg5k'], ['after 1k', 'after 4.8k'], '')

# step labels under the lower panel, rotated
for i, r in enumerate(rows):
    lab = html.escape(f"{r['step']} {r['change']}")
    yb = y_tg + PANEL_H + 14
    out.append(f'<text class="tick" transform="translate({xs[i]:.1f},{yb}) rotate(-35)" text-anchor="end">{lab}</text>')

# legend (always present for 2 series): line style + marker, so identity is not color alone
lx, ly = PAD_L + plot_w + 18, TOP + 46
for si, (name, dash) in enumerate((('1k prompt / after 1k', ''), ('4.8k prompt / after 4.8k', '6 4')), start=1):
    yy = ly + (si - 1)*20
    out.append(f'<line class="l{si}" x1="{lx}" x2="{lx + 22}" y1="{yy}" y2="{yy}"/>')
    out.append(f'<circle class="m{si}" cx="{lx + 11}" cy="{yy}" r="4"/>')
    out.append(f'<text x="{lx + 28}" y="{yy + 4}" style="font-size:11px">{name}</text>')
out.append('</svg>')
open(os.path.join(root, 'progress.svg'), 'w').write('\n'.join(out) + '\n')

# the table view, for the README
md = ['| step | change | commit | VRAM peak (MiB) | prefill 1k | prefill 4.8k | decode @1k | decode @4.8k |', '|---|---|---|---:|---:|---:|---:|---:|']
for r in rows:
    md.append(f"| {r['step']} | {r['change']} | `{r['commit']}` | {r['vram_mib']} | {r['pp1k']} | {r['pp5k']} | {r['tg1k']} | {r['tg5k']} |")
open(os.path.join(root, 'progress-table.md'), 'w').write('\n'.join(md) + '\n')
print(f'{n} steps -> progress.svg, progress-table.md')
