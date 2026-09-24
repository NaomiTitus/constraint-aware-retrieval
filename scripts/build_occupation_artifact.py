import json
d=json.load(open('reports/occupation_distribution.json'))
DATA=json.dumps(d,ensure_ascii=False,separators=(',',':'))
html = r'''<title>Occupation Tail Pruning</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Literata:opsz,wght@7..72,400;7..72,500&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
:root{
  --paper:#F4F2ED;--surface:#FBFAF7;--sunk:#EAE7DF;
  --ink:#171A23;--ink-soft:#40465A;--muted:#767D91;
  --rule:#D9D5CA;--rule-soft:#E6E3DA;
  --signal:#C25E12;--struct:#26696C;--reject:#A33C33;--accept:#3F7A3A;
  --shadow:rgba(23,26,35,.07);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#12141B;--surface:#191C25;--sunk:#0D0F15;
  --ink:#EDEAE2;--ink-soft:#B6BACA;--muted:#7E859A;
  --rule:#2A2F3D;--rule-soft:#222733;
  --signal:#E8913F;--struct:#5BA8AB;--reject:#D4685C;--accept:#7BB374;
  --shadow:rgba(0,0,0,.4);}}
:root[data-theme="dark"]{
  --paper:#12141B;--surface:#191C25;--sunk:#0D0F15;
  --ink:#EDEAE2;--ink-soft:#B6BACA;--muted:#7E859A;
  --rule:#2A2F3D;--rule-soft:#222733;
  --signal:#E8913F;--struct:#5BA8AB;--reject:#D4685C;--accept:#7BB374;
  --shadow:rgba(0,0,0,.4);}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font-family:Literata,Georgia,serif;
  font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:0 24px 80px}
h1,h2,h3,.disp{font-family:Archivo,"Helvetica Neue",sans-serif;text-wrap:balance}
h1{font-size:clamp(2rem,4.6vw,3.1rem);line-height:1.04;font-weight:700;letter-spacing:-.025em;margin:0 0 .4rem}
h2{font-size:1.32rem;font-weight:600;letter-spacing:-.015em;margin:0 0 .5rem}
p{margin:0 0 1rem}
.eyebrow{font-family:"JetBrains Mono",monospace;font-size:.7rem;letter-spacing:.16em;
  text-transform:uppercase;color:var(--muted);margin:0 0 .7rem}
header{padding:72px 0 34px;border-bottom:1px solid var(--rule)}
.lede{font-size:1.14rem;line-height:1.5;color:var(--ink-soft);max-width:58ch;margin:.9rem 0 0}
section{padding:40px 0}
code{font-family:"JetBrains Mono",monospace;font-size:.86em;background:var(--sunk);
  padding:.12em .38em;border-radius:3px}
.panel{background:var(--surface);border:1px solid var(--rule);border-radius:4px;
  box-shadow:0 1px 3px var(--shadow);overflow:hidden;margin:22px 0}
.panel-hd{padding:13px 18px;border-bottom:1px solid var(--rule-soft);display:flex;
  gap:12px;align-items:baseline;justify-content:space-between;flex-wrap:wrap}
.panel-hd .t{font-family:Archivo,sans-serif;font-weight:600;font-size:.93rem}
.panel-bd{padding:16px 18px}
canvas{display:block;width:100%;height:auto}
.ctlbar{position:sticky;top:0;z-index:5;background:var(--surface);
  border:1px solid var(--rule);border-radius:4px;padding:16px 18px;
  box-shadow:0 2px 10px var(--shadow);margin:20px 0 26px}
.ctlrow{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
input[type=range]{accent-color:var(--signal);flex:1;min-width:220px}
.ctl-label{font-family:"JetBrains Mono",monospace;font-size:.72rem;letter-spacing:.07em;
  text-transform:uppercase;color:var(--muted);white-space:nowrap}
.nval{font-family:Archivo,sans-serif;font-weight:700;font-size:1.5rem;
  font-variant-numeric:tabular-nums;letter-spacing:-.02em;min-width:3.4ch;text-align:right}
button{font-family:Archivo,sans-serif;font-weight:600;font-size:.78rem;
  background:transparent;color:var(--ink);border:1px solid var(--rule);
  padding:.4em .8em;border-radius:3px;cursor:pointer}
button:hover{border-color:var(--ink)}
button.on{background:var(--ink);color:var(--paper);border-color:var(--ink)}
button:focus-visible,input:focus-visible{outline:2px solid var(--signal);outline-offset:2px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:1px;
  background:var(--rule-soft);border:1px solid var(--rule);border-radius:4px;
  overflow:hidden;margin:20px 0}
.stats>div{background:var(--surface);padding:14px 16px}
.stats .k{font-family:"JetBrains Mono",monospace;font-size:.66rem;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted);margin-bottom:5px}
.stats .v{font-family:Archivo,sans-serif;font-size:1.5rem;font-weight:700;
  letter-spacing:-.02em;font-variant-numeric:tabular-nums;line-height:1.1}
.stats .s{font-size:.78rem;color:var(--muted);margin-top:2px}
table{width:100%;border-collapse:collapse;font-size:.88rem}
th,td{text-align:left;padding:.5em .6em;border-bottom:1px solid var(--rule-soft)}
th{font-family:Archivo,sans-serif;font-size:.68rem;letter-spacing:.08em;
  text-transform:uppercase;color:var(--muted);font-weight:600}
td.num{font-family:"JetBrains Mono",monospace;font-variant-numeric:tabular-nums;text-align:right}
tr.cut td{opacity:.35}
.scroll{overflow-x:auto;max-height:420px;overflow-y:auto}
.callout{border-left:2px solid var(--signal);padding:1px 0 1px 18px;margin:22px 0;
  color:var(--ink-soft);max-width:66ch}
.callout strong{color:var(--ink)}
.measure{max-width:66ch}
.pill{display:inline-block;font-family:"JetBrains Mono",monospace;font-size:.66rem;
  letter-spacing:.06em;text-transform:uppercase;padding:.15em .45em;border-radius:2px;
  border:1px solid currentColor;vertical-align:middle}
.pill.isco{color:var(--struct)}.pill.occ{color:var(--muted)}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-family:"JetBrains Mono",monospace;
  font-size:.7rem;color:var(--muted);padding:10px 18px;border-top:1px solid var(--rule-soft)}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}
footer{padding:34px 0 0;color:var(--muted);font-size:.85rem;border-top:1px solid var(--rule-soft)}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
<div class="wrap">
<header>
  <p class="eyebrow">ESCO occupation coverage · 10,166 Norwegian job ads</p>
  <h1>Occupation Tail Pruning</h1>
  <p class="lede">817 canonical occupation nodes, distributed with a long tail. Drag the
  cut-off to see how much of the corpus survives — and what it saves.</p>
</header>

<div class="ctlbar">
  <div class="ctlrow">
    <span class="ctl-label">Keep top</span>
    <span class="nval" id="nval">817</span>
    <input type="range" id="n" min="1" max="817" value="817">
    <span class="ctl-label">of 817</span>
    <span style="display:flex;gap:6px">
      <button data-p="50">50%</button><button data-p="75">75%</button>
      <button data-p="90">90%</button><button data-p="95">95%</button>
      <button data-p="100" class="on">all</button>
    </span>
  </div>
</div>

<div class="stats">
  <div><div class="k">Ads retained</div><div class="v" id="s-ads">9,952</div>
       <div class="s" id="s-cov">100% of tagged</div></div>
  <div><div class="k">Ads dropped</div><div class="v" id="s-drop" style="color:var(--reject)">0</div>
       <div class="s" id="s-dropp">0% of corpus</div></div>
  <div><div class="k">Extraction cost</div><div class="v" id="s-cost">$3.00</div>
       <div class="s">Haiku 4.5 batch, 30% residue</div></div>
  <div><div class="k">Embedding time</div><div class="v" id="s-emb">30m</div>
       <div class="s">nb-sbert, CPU only</div></div>
  <div><div class="k">Demo payload</div><div class="v" id="s-pay">7.6<span style="font-size:.9rem"> MB</span></div>
       <div class="s">int8 vectors + facets</div></div>
</div>

<div class="panel">
  <div class="panel-hd"><span class="t">Cumulative ad coverage</span>
    <span class="ctl-label" id="curve-note">distinct ads reached by the top N occupations</span></div>
  <div class="panel-bd"><canvas id="curve" width="1060" height="320"></canvas></div>
</div>

<div class="panel">
  <div class="panel-hd"><span class="t">Occupation distribution, ranked</span>
    <span class="ctl-label">log scale · every node shown</span></div>
  <div class="panel-bd"><canvas id="bars" width="1060" height="260"></canvas></div>
  <div class="legend">
    <span><i style="background:var(--struct)"></i>retained</span>
    <span><i style="background:var(--rule)"></i>pruned</span>
    <span><i style="background:var(--signal);width:2px;height:11px;border-radius:0"></i>cut-off</span>
  </div>
</div>

<section>
  <h2>Where the mass sits</h2>
  <div class="measure">
  <p>The distribution is steeply Zipfian. Half the corpus sits in 23 occupations; the bottom
  285 nodes carry a single ad each. That shape is what makes pruning tempting — and also what
  makes it a poor lever for cost.</p>
  </div>
  <div class="scroll"><table id="tbl"><thead><tr>
    <th>#</th><th>Occupation</th><th>Type</th><th class="num">Ads</th><th class="num">Cum. coverage</th>
  </tr></thead><tbody></tbody></table></div>
</section>

<section>
  <h2>What pruning actually buys</h2>
  <div class="callout">
  <strong>Most of the budget does not scale with corpus size.</strong> The LLM judge — the
  single largest line item at roughly <code>$11.60</code> for three passes — is a function of
  <code>pool_depth × systems × queries</code>, not of how many ads exist. Halving the corpus
  leaves it untouched.
  </div>
  <div class="measure">
  <p>Extraction is the only line that scales, and at Haiku batch rates the entire corpus costs
  about <strong>$3.00</strong>. Cutting to 75% coverage saves roughly <strong>$0.76</strong> and
  discards 2,531 ads — a bad trade for a retrieval experiment, where the discarded tail is
  precisely the vocabulary-mismatch material the graph channel exists to handle.</p>
  <p>The tail is also where the interesting personas live. <code>Sykepleier</code> at 660 ads
  needs no help from a semantic search; a single-ad occupation is exactly the query BM25 fails
  and the ESCO graph should rescue. Pruning the tail removes the evidence for the thesis.</p>
  </div>
  <div class="callout">
  <strong>Where pruning does make sense:</strong> the browser payload. The demo ships a fixed
  slice to every visitor, and there a cut-off is a page-weight decision rather than a
  scientific one — prune for the demo, keep everything for the evaluation.
  </div>
</section>

<footer>
  <p>Coverage is distinct-ad, not a sum of per-node counts — an ad may carry both an ISCO-group
  and an occupation tag. Node counts are post-canonicalisation: 903 raw codes collapse to 817
  once ISCO URI casing is normalised.</p>
</footer>
</div>
<script>
const D = __DATA__;
const OCC=D.occupations, CUM=D.cumulative, TAG=D.tagged, CORP=D.corpus, N=OCC.length;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const dpr=Math.min(devicePixelRatio||1,2);
const fmt=n=>n.toLocaleString('en-US');
function fit(c,ar){const r=c.getBoundingClientRect();c.width=r.width*dpr;
  c.height=r.width*ar*dpr;c.style.height=(r.width*ar)+'px';
  const x=c.getContext('2d');x.setTransform(dpr,0,0,dpr,0,0);return x;}
// cost model — verified pricing, Sep 2026
const PER_AD_EXTRACT = 0.30*((1411*0.50)+(120*2.50))/1e6;   // Haiku 4.5 batch, 30% residue
const JUDGE_FIXED = 11.60;
function stats(n){
  const ads=CUM[n-1], drop=TAG-ads;
  return{n,ads,drop,cov:ads/TAG,
    cost:ads*PER_AD_EXTRACT,
    emb:Math.round(ads/5000*15),
    pay:(ads*(768+700))/1048576};
}
function render(n){
  const s=stats(n);
  nval.textContent=n;
  document.getElementById('s-ads').textContent=fmt(s.ads);
  document.getElementById('s-cov').textContent=(s.cov*100).toFixed(1)+'% of tagged';
  document.getElementById('s-drop').textContent=fmt(s.drop);
  document.getElementById('s-dropp').textContent=(100*s.drop/CORP).toFixed(1)+'% of corpus';
  document.getElementById('s-cost').textContent='$'+s.cost.toFixed(2);
  document.getElementById('s-emb').textContent=s.emb+'m';
  document.getElementById('s-pay').innerHTML=s.pay.toFixed(1)+'<span style="font-size:.9rem"> MB</span>';
  drawCurve(n);drawBars(n);markTable(n);
}
function drawCurve(n){
  const c=document.getElementById('curve'),x=fit(c,.30);
  const W=c.width/dpr,H=c.height/dpr,L=52,R=14,T=14,B=28,pw=W-L-R,ph=H-T-B;
  x.clearRect(0,0,W,H);
  x.strokeStyle=css('--rule-soft');x.lineWidth=1;x.fillStyle=css('--muted');
  x.font='500 10px "JetBrains Mono",monospace';
  for(let p=0;p<=100;p+=25){const y=T+ph-(p/100)*ph;
    x.beginPath();x.moveTo(L,y);x.lineTo(W-R,y);x.stroke();
    x.fillText(p+'%',8,y+3);}
  x.beginPath();
  for(let i=0;i<N;i++){const px=L+(i/(N-1))*pw,py=T+ph-(CUM[i]/TAG)*ph;
    i?x.lineTo(px,py):x.moveTo(px,py);}
  x.strokeStyle=css('--struct');x.lineWidth=2;x.stroke();
  x.lineTo(L+pw,T+ph);x.lineTo(L,T+ph);x.closePath();
  x.fillStyle=css('--struct');x.globalAlpha=.10;x.fill();x.globalAlpha=1;
  const mx=L+((n-1)/(N-1))*pw,my=T+ph-(CUM[n-1]/TAG)*ph;
  x.strokeStyle=css('--signal');x.lineWidth=1.5;x.setLineDash([4,3]);
  x.beginPath();x.moveTo(mx,T);x.lineTo(mx,T+ph);x.stroke();x.setLineDash([]);
  x.beginPath();x.arc(mx,my,5,0,7);x.fillStyle=css('--signal');x.fill();
  x.fillStyle=css('--ink');x.font='600 11px Archivo,sans-serif';
  const lbl='top '+n+' → '+(100*CUM[n-1]/TAG).toFixed(1)+'%';
  const tw=x.measureText(lbl).width, lx=Math.min(mx+9,W-R-tw);
  x.fillText(lbl,lx,Math.max(my-10,T+12));
  x.fillStyle=css('--muted');x.font='500 10px "JetBrains Mono",monospace';
  x.fillText('1',L,H-9);x.fillText(N+' occupations',W-R-72,H-9);
}
function drawBars(n){
  const c=document.getElementById('bars'),x=fit(c,.245);
  const W=c.width/dpr,H=c.height/dpr,L=40,R=12,T=12,B=22,pw=W-L-R,ph=H-T-B;
  x.clearRect(0,0,W,H);
  const mx=Math.log10(OCC[0].ads+1),bw=pw/N;
  x.fillStyle=css('--muted');x.font='500 10px "JetBrains Mono",monospace';
  [1,10,100,660].forEach(v=>{const y=T+ph-(Math.log10(v+1)/mx)*ph;
    x.strokeStyle=css('--rule-soft');x.beginPath();x.moveTo(L,y);x.lineTo(W-R,y);x.stroke();
    x.fillStyle=css('--muted');x.fillText(v,8,y+3);});
  OCC.forEach((o,i)=>{const h=(Math.log10(o.ads+1)/mx)*ph;
    x.fillStyle=i<n?css('--struct'):css('--rule');
    x.globalAlpha=i<n?.85:.55;
    x.fillRect(L+i*bw,T+ph-h,Math.max(bw-.4,.6),h);});
  x.globalAlpha=1;
  const cx=L+n*bw;
  x.strokeStyle=css('--signal');x.lineWidth=2;
  x.beginPath();x.moveTo(cx,T-2);x.lineTo(cx,T+ph+4);x.stroke();
}
const tbody=document.querySelector('#tbl tbody');
OCC.forEach((o,i)=>{const tr=document.createElement('tr');
  tr.innerHTML='<td class="num" style="color:var(--muted)">'+(i+1)+'</td>'+
    '<td>'+o.label.replace(/[<>&]/g,'')+'</td>'+
    '<td><span class="pill '+(o.kind==='isco'?'isco">ISCO':'occ">OCC')+'</span></td>'+
    '<td class="num">'+fmt(o.ads)+'</td>'+
    '<td class="num" style="color:var(--muted)">'+(100*CUM[i]/TAG).toFixed(1)+'%</td>';
  tbody.appendChild(tr);});
const trs=[...tbody.children];
let lastN=-1;
function markTable(n){if(n===lastN)return;lastN=n;
  trs.forEach((tr,i)=>tr.classList.toggle('cut',i>=n));}
const nEl=document.getElementById('n'),nval=document.getElementById('nval');
nEl.oninput=()=>{setOn(null);render(+nEl.value);};
function setOn(b){document.querySelectorAll('[data-p]').forEach(x=>x.classList.toggle('on',x===b));}
document.querySelectorAll('[data-p]').forEach(b=>b.onclick=()=>{
  const t=+b.dataset.p/100;
  const n=t>=1?N:CUM.findIndex(v=>v/TAG>=t)+1;
  nEl.value=n;setOn(b);render(n);});
addEventListener('resize',()=>{clearTimeout(window._rt);
  window._rt=setTimeout(()=>render(+nEl.value),140);});
matchMedia('(prefers-color-scheme:dark)').addEventListener('change',()=>render(+nEl.value));
render(N);
</script>'''
open('docs/occupation-pruning.html','w').write(html.replace('__DATA__',DATA))
print('written', len(html.replace('__DATA__',DATA)), 'bytes')
