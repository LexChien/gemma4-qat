// M13: build the exact website native prompt (site-chat-v1 envelope, policy site-policy) for each eval item by importing the
// website's own read-only helpers (scripts/gemma-native-api.mjs: effectiveLocale, resolveTask/resolveResponseContract, nativeEnvelope).
// First turn of a fresh session (empty history); runtime clock fixed at 2026-11-21T02:00:00Z for reproducibility. Page locale: zh-tw for
// zh items and the 50 prompts, en for en items.  usage: node m13_envelopes.mjs <in.json> <out.json>
import fs from 'node:fs';
const W = '$WEBSITE_ROOT/scripts/';
const api = await import(W + 'gemma-native-api.mjs'); const ctx = await import(W + 'gemma-task-context.mjs');
const [inp, outp] = process.argv.slice(2); const src = JSON.parse(fs.readFileSync(inp, 'utf8')); const T = Date.parse('2026-11-21T02:00:00Z');
const out = {};
for (const [k, items] of Object.entries(src)) out[k] = items.map(([id, msg, page]) => {
  try {
    const locale = api.effectiveLocale(msg, page); const task = ctx.resolveTask(msg, [], locale);
    const rc = api.hydrateResponseContract(ctx.resolveResponseContract(msg, [], locale, task), locale);
    return { id, locale, task: task?.kind || null, contract: rc?.domain || null, prompt: api.nativeEnvelope(msg, locale, task, rc, api.runtimeContextAt(T, locale)) };
  } catch (e) { return { id, error: String(e.code || e.message) }; }
});
fs.writeFileSync(outp, JSON.stringify(out, null, 1));
