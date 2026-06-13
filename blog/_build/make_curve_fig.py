"""Build Figure 8 (toy meta-learning curve) as inline SVG and splice it into
the blog source at the <!-- FIG_CURVE --> marker."""
import json
from pathlib import Path

BUILD = Path(__file__).parent
SRC = BUILD.parent / 'discorl_deep_dive.src.html'

js = json.load(open(BUILD / 'curve.json'))
REINFORCE, UNIFORM = 0.851, 0.175
cross = next(i for i, j in enumerate(js) if j > REINFORCE)
print(f'{len(js)} points, final {js[-1]:.3f}, crosses REINFORCE at iter {cross}')

# plot geometry
X0, X1, Y0, Y1 = 70, 825, 320, 40          # data (0,0)->(X0,Y0), (600,1)->(X1,Y1)
px = lambda i: X0 + (X1 - X0) * i / 600
py = lambda j: Y0 + (Y1 - Y0) * j          # j in [0,1]

pts = ' '.join(f'{px(i):.1f},{py(j):.1f}' for i, j in enumerate(js))

xticks = ''.join(
    f'<line x1="{px(i)}" y1="{Y0}" x2="{px(i)}" y2="{Y0+5}" stroke="#6b7280"/>'
    f'<text x="{px(i)}" y="{Y0+22}" text-anchor="middle" font-size="12" fill="#6b7280">{i}</text>'
    for i in range(0, 601, 100))
yticks = ''.join(
    f'<line x1="{X0-5}" y1="{py(v)}" x2="{X0}" y2="{py(v)}" stroke="#6b7280"/>'
    f'<text x="{X0-10}" y="{py(v)+4}" text-anchor="end" font-size="12" fill="#6b7280">{v:g}</text>'
    for v in (0, 0.25, 0.5, 0.75, 1.0))

svg = f'''<figure>
<svg viewBox="0 0 860 380" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, sans-serif">
  <line x1="{X0}" y1="{Y0}" x2="{X1}" y2="{Y0}" stroke="#6b7280" stroke-width="1.2"/>
  <line x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1-10}" stroke="#6b7280" stroke-width="1.2"/>
  {xticks}
  {yticks}
  <text x="{(X0+X1)/2}" y="358" text-anchor="middle" font-size="13" fill="#6b7280">meta-iteration</text>
  <text x="20" y="{(Y0+Y1)/2}" text-anchor="middle" font-size="13" fill="#6b7280" transform="rotate(-90 20 {(Y0+Y1)/2})">J(θ₈) of a fresh agent</text>
  <line x1="{X0}" y1="{py(1.0)}" x2="{X1}" y2="{py(1.0)}" stroke="#d8d2c4" stroke-width="1.2" stroke-dasharray="3 4"/>
  <text x="{X1}" y="{py(1.0)-7}" text-anchor="end" font-size="12" fill="#9a9486">optimal policy: 1.0</text>
  <line x1="{X0}" y1="{py(REINFORCE)}" x2="{X1}" y2="{py(REINFORCE)}" stroke="#3b5ba5" stroke-width="1.5" stroke-dasharray="7 5"/>
  <text x="{X1}" y="{py(REINFORCE)+18}" text-anchor="end" font-size="12.5" fill="#3b5ba5" font-weight="600">tuned REINFORCE (best of lr sweep): 0.851</text>
  <line x1="{X0}" y1="{py(UNIFORM)}" x2="{X1}" y2="{py(UNIFORM)}" stroke="#6b7280" stroke-width="1.5" stroke-dasharray="7 5"/>
  <text x="{X1}" y="{py(UNIFORM)-8}" text-anchor="end" font-size="12.5" fill="#6b7280">uniform random policy: 0.175</text>
  <polyline points="{pts}" fill="none" stroke="#e8833a" stroke-width="2"/>
  <line x1="{px(cross)}" y1="{py(REINFORCE)}" x2="{px(cross)}" y2="{py(REINFORCE)+46}" stroke="#c0392b" stroke-width="1.2" stroke-dasharray="3 3"/>
  <text x="{px(cross)+8}" y="{py(REINFORCE)+44}" font-size="12" fill="#c0392b">passes tuned REINFORCE at iter {cross}</text>
</svg>
<figcaption><b>Figure 8 — Discovery in miniature.</b> Exact return \\(J(\\theta_8)\\) of a fresh
agent after 8 updates by the current rule, logged at every meta-iteration of the run shown in the
transcript above (single seed; the curve is raw, not smoothed — each point uses freshly sampled
trajectories, hence the flicker). The rule passes the best REINFORCE from a five-point
learning-rate sweep at iteration {cross} and saturates within ~2% of the optimal policy.
</figcaption>
</figure>'''

doc = SRC.read_text()
assert '<!-- FIG_CURVE -->' in doc
SRC.write_text(doc.replace('<!-- FIG_CURVE -->', svg, 1))
print('spliced figure into', SRC.name)
