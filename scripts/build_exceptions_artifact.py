"""Render review 002 (facet exceptions) to a self-contained page.

Reuses review 001's visual language and behaviour — db sync, keyboard, progress,
PII-scrubbed excerpts — with a different item shape and a three-way verdict.

English glosses are NOT optional here. The reviewer does not read Norwegian
(LIMITATIONS.md §1), so a page of Norwegian sentences is unreadable and the
verdicts would be guesses. The first build of this page shipped without them,
which made it useless for its only user.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "worksheet"


def main() -> None:
    items = json.loads((OUT / "exceptions_items.json").read_text(encoding="utf-8"))
    gp = OUT / "exceptions_glosses.json"
    glosses = json.loads(gp.read_text(encoding="utf-8")) if gp.exists() else {}
    missing = 0
    for it in items:
        g = glosses.get(str(it["n"])) or []
        nl = len(it["lines"])
        it["lines_en"] = g[:nl] + [None] * max(0, nl - len(g))
        ns = len(it["spans"])
        it["spans_en"] = g[nl:nl + ns]
        it["all_en"] = g[nl + ns:nl + ns + len(it.get("all_lines") or [])]
        missing += sum(1 for x in it["lines_en"] if not x)

    base = io.open(ROOT / "scripts" / "worksheet_template.html", encoding="utf-8").read()
    css = base[base.index("<style>"):base.index("</style>") + 8]

    html = _PAGE.replace("/*__CSS__*/", css).replace(
        "/*__ITEMS__*/", json.dumps(items, ensure_ascii=False))
    (OUT / "exceptions.html").write_text(html, encoding="utf-8")
    print(f"wrote {OUT/'exceptions.html'}  ({len(items)} rows, "
          f"{sum(len(i['lines']) for i in items)} lines, {missing} without a gloss)")


_PAGE = """<title>Facet Exceptions Review</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
/*__CSS__*/
<style>
  .fieldname{font-family:var(--mono);font-size:12px;color:var(--accent);
    background:var(--accent-soft);padding:2px 7px;border-radius:4px;display:inline-block}
  .claim{font-family:var(--serif);font-size:19px;margin:10px 0 4px;font-weight:600}
  .claim code{font-family:var(--mono);font-size:16px;color:var(--warn)}
  .verdicts{display:flex;gap:8px;flex-wrap:wrap}
  .vb{flex:1;min-width:150px;font:inherit;font-size:13.5px;cursor:pointer;
    padding:10px;border-radius:5px;border:1px solid var(--rule);
    background:var(--paper);color:var(--ink);text-align:left}
  .vb:hover{border-color:var(--accent)}
  .vb[aria-pressed="true"]{background:var(--accent-soft);border-color:var(--accent);
    box-shadow:inset 0 0 0 1px var(--accent)}
  .vb em{display:block;font-style:normal;font-size:11.5px;color:var(--muted);margin-top:3px}
  .noev{color:var(--warn);font-size:12.5px;margin-bottom:8px}
  .nogloss{color:var(--warn);font-size:11.5px;font-family:var(--sans);
    display:block;margin-top:6px}
</style>
<header><div class="hrow">
  <h1>Facet exceptions — are these rare values justified?</h1>
  <span class="warnbar">a wrong rare value means most of that class is wrong</span>
  <div class="prog"><span id="storenote" class="storenote">checking storage…</span>
    <span id="ptext">0 / 0</span><span class="bar"><i id="pbar"></i></span></div>
</div></header>
<div class="wrap"><nav><ol id="list"></ol></nav><main id="main"></main></div>
<script>
const ITEMS = /*__ITEMS__*/;
const KEY = "facet-exceptions-v1";
let R = {}, cur = 0, DB = null;
try { R = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { R = {}; }
const esc = s => String(s == null ? "" : s).replace(/&/g,"&amp;").replace(/</g,"&lt;")
  .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
const rec = n => (R[n] = R[n] || {verdict: null, note: ""});
const done = n => !!(R[n] && R[n].verdict);
const saveLocal = () => { try { localStorage.setItem(KEY, JSON.stringify(R)); } catch (e) {} };
const pend = {};
const save = n => {
  if (n != null && R[n]) R[n].at = new Date().toISOString();
  saveLocal();
  if (!DB || n == null) return;
  clearTimeout(pend[n]);
  pend[n] = setTimeout(() => {
    const r = R[n] || {}, it = ITEMS[n - 1];
    DB.doc(`exceptions/${n}`).set({n, uuid: it.uuid, field: it.field,
      value: it.value, verdict: r.verdict || null, note: r.note || "",
      at: r.at || new Date().toISOString()}).catch(() => {});
  }, 600);
};
function setStatus(t, w) { const e = document.getElementById("storenote");
  if (e) { e.textContent = t; e.className = "storenote" + (w ? " warn" : ""); } }
async function initDb() {
  let db = null;
  try { db = await claude.use("db"); } catch (e) { db = null; }
  if (!db) { setStatus("Saved in this browser only — not synced.", true); return; }
  DB = db;
  setStatus("Saved to this artifact.", false);
  try {
    db.collection("exceptions").onSnapshot(s => {
      let t = false;
      (s.docs || s || []).forEach(d => {
        const v = d.data ? d.data() : d;
        if (!v || v.n == null) return;
        const m = R[v.n];
        if (!m || (v.at && (!m.at || v.at > m.at))) {
          R[v.n] = {verdict: v.verdict || null, note: v.note || "", at: v.at}; t = true;
        }
      });
      if (t) { saveLocal(); renderList(); }
    }, () => setStatus("Sync interrupted — still saved locally.", true));
  } catch (e) { setStatus("Saved in this browser only.", true); }
}
function renderList() {
  const ol = document.getElementById("list"); ol.innerHTML = "";
  ITEMS.forEach((it, i) => {
    const li = document.createElement("li"), b = document.createElement("button");
    b.setAttribute("aria-current", i === cur ? "true" : "false");
    b.innerHTML = `<span class="dot ${done(it.n) ? "done" : ""}"></span>
      <span class="nm">${it.n}</span><span class="st">${esc(it.field)}</span>`;
    b.onclick = () => { cur = i; render(); };
    li.appendChild(b); ol.appendChild(li);
  });
  const n = ITEMS.filter(i => done(i.n)).length;
  document.getElementById("ptext").textContent = `${n} / ${ITEMS.length}`;
  document.getElementById("pbar").style.width = (100 * n / ITEMS.length) + "%";
}
const VERDICTS = [
  ["correct", "Justified", "the ad does say this"],
  ["wrong", "Not justified", "the ad does not support it"],
  ["unsure", "Unsure", "ambiguous — say why in the note"]];
// Norwegian first, literal English gloss under it. The gloss is what the
// verdict rests on; `må` vs `bør` and `eller` vs `og` are preserved exactly.
function line(no, en) {
  return `<div class="blk">${esc(no)}` +
    (en ? `<span class="gloss">${esc(en)}</span>`
        : `<span class="nogloss">no English gloss for this line — judge with care</span>`) +
    `</div>`;
}
function render() {
  const it = ITEMS[cur], r = rec(it.n);
  const noEv = !it.lines.length;
  document.getElementById("main").innerHTML = `
    <div class="card">
      <span class="fieldname">${esc(it.field)}</span>
      <div class="claim">The extractor says <code>${esc(it.value)}</code></div>
      <p class="why">${esc(it.about)}</p>
      <div class="meta"><span>${esc(it.title)}</span>
        <span>language level <b>${esc(it.level)}</b></span></div>
    </div>
    <div class="card">
      <h3>What the ad says that could bear on this</h3>
      ${noEv ? '<p class="noev">No sentence matched the SEARCH TERMS for this facet. '
        + 'That is a statement about the filter, not proof the ad is silent — an '
        + 'earlier version of these terms missed the evidence on 6 of 10 such rows. '
        + 'Open the full advertisement below before deciding.</p>' : ''}
      ${it.lines.length
        ? it.lines.map((l, i) => line(l, (it.lines_en || [])[i])).join("")
        : ""}
      ${it.spans.length ? `<h3 style="margin-top:14px">Its language evidence</h3>`
        + it.spans.map((s, i) => line(s, (it.spans_en || [])[i])).join("") : ""}
      <div class="navbtns" style="margin-top:12px">
        <button class="act" id="togglefull">Show the full advertisement
          (${(it.all_lines || []).length} lines)</button>
      </div>
      <div id="fulltext" hidden style="margin-top:12px">
        ${(it.all_lines || []).map((l, i) => line(l, (it.all_en || [])[i])).join("")}
      </div>
    </div>
    <div class="card">
      <h3>Your verdict</h3>
      <div class="verdicts" id="vs"></div>
      <label class="f" for="note">note</label>
      <textarea id="note">${esc(r.note)}</textarea>
      <div class="navbtns">
        <button class="act" id="prev">← previous</button>
        <button class="act primary" id="next">next →</button>
        <button class="act" id="exp" style="margin-left:auto">Show JSON</button>
      </div>
      <p class="kbd-help"><kbd>1</kbd> justified · <kbd>2</kbd> not justified ·
        <kbd>3</kbd> unsure · <kbd>j</kbd>/<kbd>k</kbd> move</p>
    </div>
    <div class="card" id="expcard" hidden>
      <h3>Results</h3><p class="hintline">Copy from here.</p>
      <textarea id="out"></textarea></div>`;
  const vs = document.getElementById("vs");
  VERDICTS.forEach(([k, lbl, hint]) => {
    const b = document.createElement("button");
    b.className = "vb"; b.type = "button";
    b.setAttribute("aria-pressed", r.verdict === k ? "true" : "false");
    b.innerHTML = `${lbl}<em>${hint}</em>`;
    b.onclick = () => { r.verdict = k; save(it.n); renderList(); render(); };
    vs.appendChild(b);
  });
  const tf = document.getElementById("togglefull");
  if (tf) tf.onclick = () => {
    const f = document.getElementById("fulltext");
    f.hidden = !f.hidden;
    tf.textContent = f.hidden
      ? `Show the full advertisement (${(it.all_lines || []).length} lines)`
      : "Hide the full advertisement";
  };
  document.getElementById("note").oninput = e => { r.note = e.target.value; save(it.n); };
  document.getElementById("prev").onclick = () => go(-1);
  document.getElementById("next").onclick = () => go(1);
  document.getElementById("exp").onclick = () => {
    const c = document.getElementById("expcard"); c.hidden = false;
    document.getElementById("out").value = JSON.stringify(
      ITEMS.filter(i => done(i.n)).map(i => ({n: i.n, uuid: i.uuid, field: i.field,
        value: i.value, verdict: R[i.n].verdict, note: (R[i.n].note || "").trim() || null})),
      null, 1);
  };
  renderList(); window.scrollTo({top: 0});
}
function go(d) { cur = Math.max(0, Math.min(ITEMS.length - 1, cur + d)); render(); }
document.addEventListener("keydown", e => {
  if (/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "j") go(1);
  if (e.key === "k") go(-1);
  const i = parseInt(e.key, 10);
  if (i >= 1 && i <= 3) { rec(ITEMS[cur].n).verdict = VERDICTS[i - 1][0];
    save(ITEMS[cur].n); renderList(); render(); }
});
render(); initDb();
</script>"""

if __name__ == "__main__":
    main()
