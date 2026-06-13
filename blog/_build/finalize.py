"""Stage 2: pygments-highlight code blocks, inline KaTeX CSS + woff2 fonts."""
import base64
import html
import re
from pathlib import Path

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

BUILD = Path(__file__).parent
SRC = BUILD / 'stage1.html'
DST = BUILD.parent / 'discorl_deep_dive.html'
KATEX_DIST = BUILD / 'node_modules' / 'katex' / 'dist'

doc = SRC.read_text()

# ---------------------------------------------------------------- pygments
formatter = HtmlFormatter(nowrap=True, style='default')
lexer = PythonLexer()
n_blocks = 0

def hl(match):
    global n_blocks
    n_blocks += 1
    code = html.unescape(match.group(1))
    return ('<pre><code class="language-python">'
            + highlight(code, lexer, formatter).rstrip('\n')
            + '</code></pre>')

doc = re.sub(
    r'<pre><code class="language-python">([\s\S]*?)</code></pre>',
    hl, doc)
print(f'highlighted {n_blocks} python blocks')

pyg_css = formatter.get_style_defs('pre code')

# ------------------------------------------------------- katex css + fonts
katex_css = (KATEX_DIST / 'katex.min.css').read_text()
n_fonts = 0

def embed_font(match):
    """Replace each @font-face src list with a single embedded woff2."""
    global n_fonts
    name = match.group(1)
    woff2 = KATEX_DIST / 'fonts' / f'{name}.woff2'
    b64 = base64.b64encode(woff2.read_bytes()).decode()
    n_fonts += 1
    return f'src:url(data:font/woff2;base64,{b64}) format("woff2")'

katex_css = re.sub(
    r'src:url\(fonts/(KaTeX_[A-Za-z0-9-]+)\.woff2\)[^;}]*',
    embed_font, katex_css)
assert 'url(fonts/' not in katex_css, 'unembedded font url remains'
print(f'embedded {n_fonts} woff2 fonts')

# ------------------------------------------------------------------ inject
assets = ('<style>\n' + katex_css + '\n</style>\n'
          '<style>\n' + pyg_css + '\n</style>')
assert '<!-- STATIC_ASSETS -->' in doc
doc = doc.replace('<!-- STATIC_ASSETS -->', assets, 1)

# ------------------------------------------------------------------ checks
body = doc[doc.index('<body>'):]
assert '$$' not in body, 'unrendered display math remains'
assert re.search(r'\\\(', body) is None, 'unrendered inline math remains'
DST.write_text(doc)
print(f'wrote {DST} ({len(doc)/1e6:.2f} MB)')
