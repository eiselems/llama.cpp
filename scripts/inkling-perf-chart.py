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

W, PAD_L, PAD_R = 960, 64, 120
PANEL_H, GAP, TOP = 220, 58, 78
X_LAB = 70                                   # room for the rotated step labels under the lower panel
H = TOP + 30 + 2*PANEL_H + GAP + X_LAB
plot_w = W - PAD_L - PAD_R
n = len(rows)
xs = [PAD_L + (plot_w*(i + 0.5)/n) for i in range(n)]

# palette: categorical slots 1-2, light + dark (validated: CVD dE >= 24 in both modes)
CSS = """
.bg { fill: #fcfcfb; }
text { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; fill: #52514e; font-size: 12px; }
.title { fill: #0b0b0b; font-size: 15px; font-weight: 600; }
.ptitle { fill: #0b0b0b; font-size: 13px; font-weight: 600; }
.tick { fill: #8a8984; font-size: 11px; }
.grid { stroke: #e6e5e0; stroke-width: 1; }
.base { stroke: #8a8984; stroke-width: 1; }
.l1 { stroke: #2a78d6; fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.l2 { stroke: #eb6834; fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; stroke-dasharray: 6 4; }
.m1 { fill: #2a78d6; stroke: #fcfcfb; stroke-width: 2; }
.m2 { fill: #eb6834; stroke: #fcfcfb; stroke-width: 2; }
.val { fill: #0b0b0b; font-size: 11px; font-weight: 600; }
@media (prefers-color-scheme: dark) {
  .bg { fill: #1a1a19; }
  text { fill: #c3c2b7; }
  .title, .ptitle, .val { fill: #f0efec; }
  .tick { fill: #8f8e88; }
  .grid { stroke: #33332f; }
  .base { stroke: #8f8e88; }
  .l1 { stroke: #3987e5; } .l2 { stroke: #d95926; }
  .m1 { fill: #3987e5; stroke: #1a1a19; } .m2 { fill: #d95926; stroke: #1a1a19; }
}
"""

def nice_max(v):
    for mag in (1, 10, 100, 1000, 10000):
        for step in (1, 1.25, 1.5, 2, 2.5, 5):
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
       f'<style>{CSS}</style>', f'<rect class="bg" width="{W}" height="{H}"/>',
       f'<text class="title" x="{PAD_L}" y="26">Inkling-2-Mini on RTX 3070 8 GB - llama.cpp fork, per optimization step</text>',
       f'<text x="{PAD_L}" y="44">tokens/s, higher is better. UD-Q4_K_M, 32k ctx, q8_0 KV (q4_0 from S10). From B10: bench.py, decode = median of 6 prompts, 2nd decode line = after a 30k prompt.</text>']

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
        other = [f(rows[last][k2]) for k2 in keys if k2 != k and f(rows[last][k2]) is not None]
        above = not other or lv >= other[0]
        out.append(f'<text class="val" x="{xs[last] + 10:.1f}" y="{y(lv) + (-6 if above else 14):.1f}">{lv:g}</text>')
    # first-to-last gain for series 1
    first = next(f(r[keys[0]]) for r in rows if f(r[keys[0]]) is not None)
    lastv = [f(r[keys[0]]) for r in rows if f(r[keys[0]]) is not None][-1]
    out.append(f'<text class="tick" x="{PAD_L + plot_w}" y="{y0 - 10}" text-anchor="end">{names[0]}: {first:g} -> {lastv:g} ({lastv/first:.1f}x){unit_note}</text>')

y_pp = TOP + 30
panel(y_pp, 'Prefill (prompt processing)', ['pp1k', 'pp5k'], ['1k prompt', '4.8k prompt'], '')
y_tg = y_pp + PANEL_H + GAP
panel(y_tg, 'Decode (tokens/s; S rows: 128 tokens after 1k / 4.8k prompts; B/E rows: see subtitle)', ['tg1k', 'tg5k'], ['after 1k', 'after 4.8k'], '')

# step labels under the lower panel, rotated
import textwrap
col_w = plot_w/n
chars = max(10, int(col_w/6.2))
for i, r in enumerate(rows):
    yb = y_tg + PANEL_H + 16
    out.append(f'<text class="ptitle" x="{xs[i]:.1f}" y="{yb}" text-anchor="middle" style="font-size:12px">{html.escape(r["step"])}</text>')
    for j, line in enumerate(textwrap.wrap(r['change'], chars)[:3]):
        out.append(f'<text class="tick" x="{xs[i]:.1f}" y="{yb + 14 + 13*j}" text-anchor="middle">{html.escape(line)}</text>')

# legend (always present for 2 series): line style + marker, so identity is not color alone
lx, ly = PAD_L, 64
for name in ('1k-token prompt', '4.8k-token prompt'):
    si = 1 if name.startswith('1k') else 2
    out.append(f'<line class="l{si}" x1="{lx}" x2="{lx + 22}" y1="{ly}" y2="{ly}"/>')
    out.append(f'<circle class="m{si}" cx="{lx + 11}" cy="{ly}" r="4"/>')
    out.append(f'<text x="{lx + 28}" y="{ly + 4}">{name}</text>')
    lx += 28 + 7*len(name) + 24
out.append('</svg>')
open(os.path.join(root, 'progress.svg'), 'w').write('\n'.join(out) + '\n')

# the table view, for the README
md = ['| step | change | commit | VRAM peak (MiB) | prefill 1k | prefill 4.8k | decode @1k | decode @4.8k |', '|---|---|---|---:|---:|---:|---:|---:|']
for r in rows:
    md.append(f"| {r['step']} | {r['change']} | `{r['commit']}` | {r['vram_mib']} | {r['pp1k']} | {r['pp5k']} | {r['tg1k']} | {r['tg5k']} |")
open(os.path.join(root, 'progress-table.md'), 'w').write('\n'.join(md) + '\n')
print(f'{n} steps -> progress.svg, progress-table.md')
