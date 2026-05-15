import katex from 'katex';
import fs from 'node:fs';
import path from 'node:path';

const dataDir = process.argv[2];
const files = fs.readdirSync(dataDir).filter(f => f.endsWith('.json'));

const errors = [];

function extractMath(text) {
  if (!text) return [];
  const out = [];
  // Block math $$...$$
  const blockRe = /\$\$([\s\S]+?)\$\$/g;
  let m;
  while ((m = blockRe.exec(text)) !== null) {
    out.push({ kind: 'block', expr: m[1].trim() });
  }
  // Strip block math first, then find inline
  const stripped = text.replace(blockRe, '');
  const inlineRe = /\$([^$\n]+?)\$/g;
  while ((m = inlineRe.exec(stripped)) !== null) {
    out.push({ kind: 'inline', expr: m[1].trim() });
  }
  return out;
}

function checkText(text, location) {
  const exprs = extractMath(text);
  for (const { kind, expr } of exprs) {
    try {
      katex.renderToString(expr, {
        throwOnError: true,
        displayMode: kind === 'block',
        strict: 'error',
      });
    } catch (err) {
      errors.push({
        location,
        kind,
        expr: expr.length > 100 ? expr.slice(0, 100) + '...' : expr,
        err: err.message.split('\n')[0].slice(0, 200),
      });
    }
  }
}

function walk(member, slug, path = []) {
  const name = member.name || '?';
  const loc = `${slug}/${path.join('/')}${path.length ? '/' : ''}${name}`;
  checkText(member.summary, `${loc}:summary`);
  checkText(member.extended, `${loc}:extended`);
  for (const note of member.notes || []) checkText(note, `${loc}:notes`);
  for (const ex of member.examples || []) checkText(ex, `${loc}:examples`);
  for (const p of member.parameters || []) checkText(p.description, `${loc}:param[${p.name}]`);
  for (const r of member.raises || []) checkText(r.description, `${loc}:raises`);
  if (member.returns?.description) checkText(member.returns.description, `${loc}:returns`);
  for (const a of member.attributes || []) checkText(a.description, `${loc}:attr[${a.name}]`);
  // class methods
  for (const meth of member.methods || []) walk(meth, slug, [...path, name]);
}

for (const file of files.sort()) {
  const data = JSON.parse(fs.readFileSync(path.join(dataDir, file), 'utf8'));
  const slug = data.slug || file.replace('.json', '');
  // module-level
  checkText(data.summary, `${slug}:summary`);
  checkText(data.extended, `${slug}:extended`);
  for (const m of data.members || []) walk(m, slug);
  // class-module: methods at top
  for (const m of data.methods || []) walk(m, slug);
}

console.log(`Total errors: ${errors.length}`);
if (errors.length > 0) {
  // Group by error message head
  const byErr = {};
  for (const e of errors) {
    const key = e.err.slice(0, 80);
    (byErr[key] ||= []).push(e);
  }
  console.log('\n--- by error type ---');
  for (const [key, items] of Object.entries(byErr).sort((a, b) => b[1].length - a[1].length)) {
    console.log(`\n[${items.length}] ${key}`);
    for (const e of items.slice(0, 3)) {
      console.log(`    @${e.location}  (${e.kind})`);
      console.log(`     expr: ${e.expr}`);
    }
  }
}
