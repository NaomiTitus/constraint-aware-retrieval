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
        nl, ns = len(it["lines"]), len(it["spans"])
        na = len(it.get("all_lines") or [])
        want = 1 + nl + ns + na          # title, then lines, spans, full ad
        if g and len(g) != want:
            # A LENGTH MISMATCH IS NOT RECOVERABLE BY SLICING. Every gloss after
            # the short list shifts onto the wrong sentence, and the page then
            # shows confident English under Norwegian that does not say it —
            # which is the worst thing this page can do to a reviewer who cannot
            # read the original. Drop the whole item's glosses and say so.
            print(f"   row {it['n']}: gloss list is {len(g)}, expected {want} "
                  f"— dropped rather than misaligned; re-run scripts/gloss.py")
            g = []
        it["title_en"] = g[0] if g else None
        it["lines_en"] = g[1:1 + nl] if g else [None] * nl
        it["spans_en"] = g[1 + nl:1 + nl + ns] if g else [None] * ns
        it["all_en"] = g[1 + nl + ns:] if g else [None] * na
        missing += sum(1 for x in it["lines_en"] if not x)

    base = io.open(ROOT / "scripts" / "worksheet_template.html", encoding="utf-8").read()
    css = base[base.index("<style>"):base.index("</style>") + 8]

    css = css.replace(".blk{", ".blk{cursor:auto;")
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
  .illegal{border:1px solid var(--warn);background:var(--warn-soft,transparent);
    border-radius:5px;padding:9px 11px;margin:10px 0;font-size:12.5px;color:var(--warn)}
  .illegal b{font-family:var(--mono)}
  .qlabel{font-family:var(--sans);font-size:13px;font-weight:600;margin:14px 0 6px}
  .qlabel span{display:block;font-weight:400;font-size:11.5px;color:var(--muted);
    margin-top:2px}
  .evcount{font-size:11.5px;color:var(--muted);margin:0 0 9px}
  .spanhead{font-size:12.5px;color:var(--muted);margin:16px 0 6px;
    font-family:var(--sans)}
  .titlegloss{color:var(--muted);font-style:italic}
  .carried{border:1px solid var(--accent);border-radius:5px;padding:9px 11px;
    margin:10px 0;font-size:12.5px}
  .carried b{font-family:var(--mono)}
  .carried .why{display:block;margin-top:5px;color:var(--warn)}
  .vb.suggested{border-style:dashed}
  .vb.suggested::after{content:" · your earlier answer";font-size:10.5px;
    color:var(--muted)}
  .dot.carried-dot{box-shadow:inset 0 0 0 2px var(--accent)}
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
const KEY = "facet-exceptions-v2";  // v1 held single-verdict records
let R = {}, cur = 0, DB = null;
try { R = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { R = {}; }
// Seed carried verdicts BEFORE the first render, so the sidebar and the
// progress counter show them from the start rather than only once each row is
// opened. A stored record always wins — the reviewer's later answer beats the
// carried one it was derived from.
// A stored record is keyed by ROW NUMBER, and row numbers are assigned by
// sorting on (field, uuid) — so if the exception set ever changes, row 14
// becomes a different advertisement and a stored answer silently follows the
// number onto an ad it was never about. Verified by hand that the numbering
// did not move this time; not a reason to leave it unguarded. Every record now
// carries the uuid it was given for, and one that does not match is dropped.
ITEMS.forEach(it => {
  const r = R[it.n];
  if (r && r.uuid && r.uuid !== it.uuid) delete R[it.n];
});
ITEMS.forEach(it => {
  if (it.carried && !R[it.n]) {
    R[it.n] = {uuid: it.uuid, fact: it.carried.fact,
               value_ok: it.carried.value_ok, note: it.carried.note || "",
               carried: true, confirmed: false};
  }
});
const esc = s => String(s == null ? "" : s).replace(/&/g,"&amp;").replace(/</g,"&lt;")
  .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
// TWO questions, not one. The first build asked "is this justified?" and the
// honest answer on three rows was yes-to-the-fact and no-to-the-value: the ad
// really did state a documentation-language rule, and the value recorded for it
// was borrowed from another field's enum. One button could not say that.
// A carried verdict is PREFILLED but NOT DONE. reviews/README.md: a suggested
// value is a suggestion, and every value stays unconfirmed until touched. That
// rule was written for machine prefill; it applies at least as strongly here,
// because some of these answers were given while the page was hiding evidence.
function rec(n) {
  if (!R[n]) {
    const it = ITEMS[n - 1], c = it && it.carried;
    R[n] = c ? {fact: c.fact, value_ok: c.value_ok, note: c.note || "",
                carried: true, confirmed: false}
             : {fact: null, value_ok: null, note: ""};
  }
  return R[n];
}
// Whether a row is CARRIED is a property of the row, not of whatever happens
// to be in local storage. Reading it from the stored record meant an answer
// saved in an earlier pass — which has no `carried` flag — counted as finished,
// and the confirmation step this whole rebuild exists for was skipped without
// a word. The item is the authority; `confirmed` is the only thing the record
// gets to say.
const done = n => {
  const r = R[n], it = ITEMS[n - 1];
  if (!r || !r.fact || !r.value_ok) return false;
  return (it && it.carried) ? !!r.confirmed : true;
};
const saveLocal = () => { try { localStorage.setItem(KEY, JSON.stringify(R)); } catch (e) {} };
const pend = {};
const save = n => {
  if (n != null && R[n]) {
    R[n].at = new Date().toISOString();
    R[n].uuid = (ITEMS[n - 1] || {}).uuid;   // so a renumbering cannot steal it
  }
  saveLocal();
  if (!DB || n == null) return;
  clearTimeout(pend[n]);
  pend[n] = setTimeout(() => {
    const r = R[n] || {}, it = ITEMS[n - 1];
    DB.doc(`exceptions_v2/${n}`).set({n, uuid: it.uuid, field: it.field,
      value: it.value, fact: r.fact || null, value_ok: r.value_ok || null,
      note: r.note || "", carried: !!r.carried, confirmed: !!r.confirmed,
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
    db.collection("exceptions_v2").onSnapshot(s => {
      let t = false;
      (s.docs || s || []).forEach(d => {
        const v = d.data ? d.data() : d;
        if (!v || v.n == null) return;
        const m = R[v.n];
        if (v.at && (!m || !m.at || v.at > m.at)) {
          R[v.n] = {fact: v.fact || null, value_ok: v.value_ok || null,
                    note: v.note || "", carried: !!v.carried,
                    confirmed: !!v.confirmed, at: v.at}; t = true;
        }
      });
      if (t) { saveLocal(); render(); }   // render() also renders the list
    }, () => setStatus("Sync interrupted — still saved locally.", true));
  } catch (e) { setStatus("Saved in this browser only.", true); }
}
function renderList() {
  const ol = document.getElementById("list"); ol.innerHTML = "";
  ITEMS.forEach((it, i) => {
    const li = document.createElement("li"), b = document.createElement("button");
    b.setAttribute("aria-current", i === cur ? "true" : "false");
    const rr = R[it.n];
    const pend = rr && rr.carried && !rr.confirmed;
    b.innerHTML = `<span class="dot ${done(it.n) ? "done" : ""}${pend ? " carried-dot" : ""}"></span>
      <span class="nm">${it.n}</span><span class="st">${esc(it.field)}</span>`;
    b.onclick = () => { cur = i; render(); };
    li.appendChild(b); ol.appendChild(li);
  });
  const n = ITEMS.filter(i => done(i.n)).length;
  const pending = ITEMS.filter(i => i.carried && !(R[i.n] && R[i.n].confirmed)).length;
  document.getElementById("ptext").textContent =
    `${n} / ${ITEMS.length}` + (pending ? ` · ${pending} to confirm` : "");
  document.getElementById("pbar").style.width = (100 * n / ITEMS.length) + "%";
}
const WHY = {"value_changed": "The extractor now records a DIFFERENT value than the one you judged, so your earlier answer was about something else.", "tool_was_wrong": "This page was at fault on this row when you judged it \u2014 evidence was hidden by a filter cap or a vocabulary gap. What you were shown was incomplete.", "note_disagrees": "Your note and your buttons disagree. Probably a mis-click; yours to resolve."};
// Q1 is about the AD. Q2 is about the RECORD. They came apart on three rows:
// the ad genuinely stated a documentation-language rule (Q1 yes) and the value
// written down was borrowed from another field's enum (Q2 no). A single
// "Justified?" button forced those into one answer and got the wrong one.
const Q1 = [["yes", "Yes — the ad says it", "the fact is in the text"],
            ["no", "No — the ad does not", "nothing in the ad supports it"],
            ["unsure", "Unsure", "say why in the note"]];
// A row whose value is the SILENT one asserts nothing, so "is the fact in the
// ad" has no answer — both buttons read wrong. The useful question for these
// is the false-negative one: did the extractor just lose something real? The
// answer that SUPPORTS the extraction is "No" in both wordings, which keeps
// the two sets of answers comparable when they are counted later.
const Q1_SILENT = [
  ["yes", "Yes — the ad DOES state something", "the extractor has missed it"],
  ["no", "No — the ad is silent", "nothing here to record"],
  ["unsure", "Unsure", "say why in the note"]];
const Q2 = [["yes", "Yes — right field, right value", ""],
            ["no", "No — wrong field or wrong value", "e.g. belongs elsewhere"],
            ["unsure", "Unsure", "say why in the note"]];
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
      ${it.value_is_legal === false ? `<div class="illegal">
        <b>${esc(it.value)}</b> is not one of this field's legal values. It is
        borrowed from a neighbouring field's enum, so the record is discarded
        downstream however well the ad supports the underlying fact.
        Legal values: ${(it.legal_values || []).map(esc).join(" · ")}.
        <br>Answer question 1 about the AD and question 2 about the VALUE —
        they can honestly differ here.</div>` : ""}
      ${it.value_v8 != null && String(it.value_v8) !== String(it.value)
        ? `<p class="evcount">Changed by the prompt fix: this ad recorded
             <code>${esc(it.value_v8)}</code> before, <code>${esc(it.value)}</code>
             now. You are judging the new value.</p>` : ""}
      <p class="why">${esc(it.about)}</p>
      <div class="meta"><span>${esc(it.title)}${it.title_en
          ? ` <span class="titlegloss">— ${esc(it.title_en)}</span>` : ""}</span>
        <span>language level <b>${esc(it.level)}</b></span></div>
    </div>
    <div class="card">
      <h3>What the ad says that could bear on this</h3>
      ${noEv ? '<p class="noev">No sentence matched the SEARCH TERMS for this facet. '
        + 'That is a statement about the filter, not proof the ad is silent — an '
        + 'earlier version of these terms missed the evidence on 6 of 10 such rows. '
        + 'Open the full advertisement below before deciding.</p>'
        : `<p class="evcount">${it.lines.length} of ${(it.all_lines || []).length}
             lines in the ad matched these search terms. All of them are shown —
             an earlier build silently capped this at 8 and dropped short lines,
             which hid the decisive sentence on two rows.</p>`}
      ${it.lines.length
        ? it.lines.map((l, i) => line(l, (it.lines_en || [])[i])).join("")
        : ""}
      ${it.spans.length ? `<p class="spanhead">Separately — the spans the model
        quoted for the LANGUAGE verdict. Usually nothing to do with
        <code>${esc(it.field)}</code>; shown because they are the only text the
        model committed to.</p>`
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
      ${it.carried ? `<div class="carried">
        Carried over from your earlier pass
        ${it.carried.at ? `(${esc(it.carried.at.slice(11, 16))})` : ""} —
        you judged <b>${esc(it.carried.value_judged)}</b> as
        <b>${esc(it.carried.fact)}</b> / <b>${esc(it.carried.value_ok)}</b>.
        The answers below are filled in with it. <b>It does not count until you
        confirm.</b>
        ${(it.carried.flags || []).map(f => `<span class="why">${esc(WHY[f] || f)}</span>`).join("")}
        <div class="navbtns" style="margin-top:8px">
          <button class="act primary" id="confirmcarry">Confirm — unchanged</button>
        </div></div>` : ""}
      <p class="qlabel">${it.silent_value
        ? `1 · Does the advertisement state anything about
             <code>${esc(it.field)}</code>?
           <span>This row records SILENCE, so there is no claim to check. The
             question is the other way round: is there something in the ad the
             extractor has missed?</span>`
        : `1 · Is the fact in the advertisement?
           <span>Ignore which field it was filed under — just: does the ad say
             it?</span>`}</p>
      <div class="verdicts" id="vs1"></div>
      <p class="qlabel">2 · Is <code>${esc(it.value)}</code> the right value for
        <code>${esc(it.field)}</code>?
        <span>${it.value_is_legal === false
          ? "This value is NOT one of this field's legal values — see the warning above."
          : "Legal values: " + ((it.legal_values || ["(free-form)"]).join(" · "))}</span></p>
      <div class="verdicts" id="vs2"></div>
      <label class="f" for="note">note</label>
      <textarea id="note">${esc(r.note)}</textarea>
      <div class="navbtns">
        <button class="act" id="prev">← previous</button>
        <button class="act primary" id="next">next →</button>
        <button class="act" id="exp" style="margin-left:auto">Show JSON</button>
      </div>
      <p class="kbd-help">ad: <kbd>1</kbd> yes · <kbd>2</kbd> no · <kbd>3</kbd> unsure &nbsp;·&nbsp;
        value: <kbd>4</kbd> yes · <kbd>5</kbd> no · <kbd>6</kbd> unsure &nbsp;·&nbsp;
        <kbd>j</kbd>/<kbd>k</kbd> move</p>
    </div>
    <div class="card" id="expcard" hidden>
      <h3>Results</h3><p class="hintline">Copy from here.</p>
      <textarea id="out"></textarea></div>`;
  [["vs1", it.silent_value ? Q1_SILENT : Q1, "fact"],
   ["vs2", Q2, "value_ok"]].forEach(([id, opts, key]) => {
    const host = document.getElementById(id);
    opts.forEach(([k, lbl, hint]) => {
      const b = document.createElement("button");
      b.className = "vb"; b.type = "button";
      const cur = rec(it.n);
      b.setAttribute("aria-pressed", cur[key] === k ? "true" : "false");
      if (cur.carried && !cur.confirmed && cur[key] === k) b.classList.add("suggested");
      b.innerHTML = `${lbl}${hint ? `<em>${hint}</em>` : ""}`;
      // read the record FRESH: a remote snapshot replaces R[n] with a NEW
      // object, and a handler closing over the old one wrote into an orphan
      // that saveLocal() never serialised. Silent loss of an edit.
      b.onclick = () => {
        const r = rec(it.n);
        r[key] = k;
        // Touching an answer IS the confirmation — the reviewer has now looked.
        if (r.carried) r.confirmed = true;
        save(it.n); render();
      };
      host.appendChild(b);
    });
  });
  const cc = document.getElementById("confirmcarry");
  if (cc) cc.onclick = () => {
    const r = rec(it.n); r.confirmed = true; save(it.n); go(1);
  };
  const tf = document.getElementById("togglefull");
  if (tf) tf.onclick = () => {
    const f = document.getElementById("fulltext");
    f.hidden = !f.hidden;
    tf.textContent = f.hidden
      ? `Show the full advertisement (${(it.all_lines || []).length} lines)`
      : "Hide the full advertisement";
  };
  document.getElementById("note").oninput = e => { rec(it.n).note = e.target.value; save(it.n); };
  document.getElementById("prev").onclick = () => go(-1);
  document.getElementById("next").onclick = () => go(1);
  document.getElementById("exp").onclick = () => {
    const c = document.getElementById("expcard"); c.hidden = false;
    document.getElementById("out").value = JSON.stringify(
      ITEMS.filter(i => done(i.n)).map(i => ({n: i.n, uuid: i.uuid, field: i.field,
        value: i.value, value_was: i.value_v8 ?? null,
        fact_in_ad: R[i.n].fact, value_correct: R[i.n].value_ok,
        note: (R[i.n].note || "").trim() || null})),
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
  if (i >= 1 && i <= 3) { rec(ITEMS[cur].n).fact = Q1[i - 1][0];   // same keys
    save(ITEMS[cur].n); render(); }
  if (i >= 4 && i <= 6) { rec(ITEMS[cur].n).value_ok = Q2[i - 4][0];
    save(ITEMS[cur].n); render(); }
});
render(); initDb();
</script>"""

if __name__ == "__main__":
    main()
