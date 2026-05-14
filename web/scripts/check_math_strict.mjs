import katex from 'katex';
import fs from 'node:fs';
import path from 'node:path';

const dataDir = process.argv[2];
const files = fs.readdirSync(dataDir).filter(f => f.endsWith('.json'));

const warnings = [];

function extractMath(text) {
  if (!text) return [];
  const out = [];
  const blockRe = /\$\$([\s\S]+?)\$\$/g;
  let m;
  while ((m = blockRe.exec(text)) !== null) out.push({ kind: 'block', expr: m[1].trim() });
  const stripped = text.replace(blockRe, '');
  const inlineRe = /\$([^$\n]+?)\$/g;
  while ((m = inlineRe.exec(stripped)) !== null) out.push({ kind: 'inline', expr: m[1].trim() });
  return out;
}

function checkText(text, location) {
  for (const { kind, expr } of extractMath(text)) {
    let warned = false;
    try {
      katex.renderToString(expr, {
        throwOnError: true,
        displayMode: kind === 'block',
        strict: (errCode, errMsg) => {
          warnings.push({ location, kind, expr: expr.slice(0, 80), code: errCode, msg: errMsg.slice(0, 150) });
          warned = true;
          return 'ignore';
        },
      });
    } catch (err) {
      // shouldn't reach since strict callback returns 'ignore'
    }
  }
}

function walk(member, slug, p = []) {
  const loc = `${slug}/${p.join('/')}${p.length ? '/' : ''}${member.name || '?'}`;
  checkText(member.summary, `${loc}:summary`);
  checkText(member.extended, `${loc}:extended`);
  for (const n of member.notes || []) checkText(n, `${loc}:notes`);
  for (const e of member.examples || []) checkText(e, `${loc}:examples`);
  for (const pr of member.parameters || []) checkText(pr.description, `${loc}:param`);
  for (const r of member.raises || []) checkText(r.description, `${loc}:raises`);
  if (member.returns?.description) checkText(member.returns.description, `${loc}:returns`);
  for (const a of member.attributes || []) checkText(a.description, `${loc}:attr`);
  for (const meth of member.methods || []) walk(meth, slug, [...p, member.name || '?']);
}

for (const file of files.sort()) {
  const data = JSON.parse(fs.readFileSync(path.join(dataDir, file), 'utf8'));
  const slug = data.slug || file.replace('.json', '');
  checkText(data.summary, `${slug}:summary`);
  checkText(data.extended, `${slug}:extended`);
  for (const m of data.members || []) walk(m, slug);
  for (const m of data.methods || []) walk(m, slug);
}

console.log(`Total warnings: ${warnings.length}`);
const byCode = {};
for (const w of warnings) (byCode[w.code] ||= []).push(w);
for (const [code, items] of Object.entries(byCode).sort((a, b) => b[1].length - a[1].length)) {
  console.log(`\n[${items.length}] ${code}`);
  for (const w of items.slice(0, 3)) {
    console.log(`  @${w.location}`);
    console.log(`    expr: ${w.expr}`);
    console.log(`    msg:  ${w.msg}`);
  }
}
