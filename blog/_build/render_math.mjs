import katex from './node_modules/katex/dist/katex.mjs';
import { readFileSync, writeFileSync } from 'fs';

let src = readFileSync(process.argv[2], 'utf8');

// Drop the CDN <link>/<script> block (KaTeX + highlight.js); the python pass
// injects inlined static CSS at this marker instead.
const cdnBlock = /<!-- KaTeX -->[\s\S]*?hljs\.highlightAll\(\)\);<\/script>/;
if (!cdnBlock.test(src)) {
  console.error('CDN block not found');
  process.exit(1);
}
src = src.replace(cdnBlock, '<!-- STATIC_ASSETS -->');

const unescape = (s) =>
  s.replaceAll('&amp;', '&').replaceAll('&lt;', '<').replaceAll('&gt;', '>');

let nDisplay = 0, nInline = 0;
const render = (tex, displayMode) => {
  try {
    return katex.renderToString(unescape(tex), {
      displayMode,
      throwOnError: true,
      strict: 'ignore',
    });
  } catch (e) {
    console.error(`KATEX ERROR (${displayMode ? 'display' : 'inline'}):`);
    console.error(tex.slice(0, 200));
    console.error(String(e).slice(0, 300));
    process.exit(1);
  }
};

let out = src.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => {
  nDisplay++;
  return render(tex, true);
});
out = out.replace(/\\\(([\s\S]+?)\\\)/g, (_, tex) => {
  nInline++;
  return render(tex, false);
});

writeFileSync(process.argv[3], out);
console.log(`rendered ${nDisplay} display + ${nInline} inline formulas`);
