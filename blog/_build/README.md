# Build pipeline for discorl_deep_dive.html

Edit `../discorl_deep_dive.src.html` (math as $$...$$ / \(...\), code as
plain `<pre><code class="language-python">`), then rebuild:

    node render_math.mjs ../discorl_deep_dive.src.html stage1.html
    ../../.venv/bin/python finalize.py

Output `../discorl_deep_dive.html` is fully static: KaTeX pre-rendered,
Pygments-highlighted code, woff2 fonts embedded as data URIs. No JS, no CDN.
