// Latency benchmark for the shipped page's own score(), so the README's numbers have a
// source. Runs the real index through the real scoring function — no mocks, no subset.
// The point is that the figure in the README is reproducible, not that it is flattering.
import fs from "fs";
const ROOT = new URL("..", import.meta.url).pathname;
const html = fs.readFileSync(ROOT + "docs/index.html", "utf8");
const el = () => ({innerHTML:"",textContent:"",value:"",hidden:true,disabled:false,
                   children:[],addEventListener(){},dataset:{}});
globalThis.document = {getElementById: el, querySelectorAll: () => []};
globalThis.fetch = async () => ({ok:true, json: async () => ({})});

const t0 = Date.now();
const idx = JSON.parse(fs.readFileSync(ROOT + "docs/data/index.json", "utf8"));
const gaz = JSON.parse(fs.readFileSync(ROOT + "docs/data/occupations.json", "utf8"));
const tParse = Date.now() - t0;

const j = html.slice(html.indexOf("<script>") + 8, html.lastIndexOf("</script>"));
const m = new Function(j.replace(/\(async\(\)=>\{[\s\S]*$/, "") + `;return {parse,score,normPlace,
 set:(a,g)=>{ADS=a;GAZ=g.labels;TOKENS=g.tokens;TITLES=g.titles||{};NDOC=ADS.length;
  GAZKEYS=Object.keys(GAZ).sort((x,y)=>y.length-x.length);
  for(const ad of ADS){const s=new Set(norm(ad.t+" "+ad.o+" "+ad.e+" "+(ad.k||[]).join(" ")+" "+(ad.g||[]).join(" ")).split(" "));
    for(const w of s) if(w) DF[w]=(DF[w]||0)+1;}
  for(const ad of ADS)for(const [q,c] of (ad.L||[])){const nm=normPlace(q),nc=normPlace(c);
    if(nc)COUNTIES.add(nc); if(nm&&!(nm in MUNI))MUNI[nm]=nc||"";}}};`)();
const t1 = Date.now(); m.set(idx.ads, gaz); const tIndex = Date.now() - t1;

const QUERIES = [
  "I am a nurse in Bergen. I do not speak norwegian.",
  "electrical engineer, low voltage, automation",
  "AI engineer, 5 years python, I don t speak norwegian",
  "barnehagelærer i Oslo, fast stilling",
  "upper secondary school teacher, I only speak english",
  "truck driver, heavy vehicle licence",
  "sykepleier med autorisasjon, Trondheim",
  "warehouse worker, no norwegian, Oslo",
];
const N = Number(process.argv[2] || 400);
// Warm up so the first-call bigram DF cache is not charged to the measurement.
for (const q of QUERIES) m.score(m.parse(q), 0.7);

const t = [];
for (let i = 0; i < N; i++) {
  const q = QUERIES[i % QUERIES.length];
  const a = process.hrtime.bigint();
  m.score(m.parse(q), 0.7).slice(0, 10);
  t.push(Number(process.hrtime.bigint() - a) / 1e6);
}
t.sort((a, b) => a - b);
const p = x => t[Math.min(t.length - 1, Math.floor(t.length * x))];
const gz = "see `gzip -9c docs/data/index.json | wc -c`";
console.log(`node        ${process.version}   ${process.platform}/${process.arch}`);
console.log(`ads         ${idx.ads.length}   queries ${QUERIES.length}   runs ${N}`);
console.log(`JSON.parse  ${tParse} ms      (index + gazetteer, one-off)`);
console.log(`index build ${tIndex} ms      (DF + place tables, one-off)`);
console.log(`query  p50  ${p(0.50).toFixed(1)} ms`);
console.log(`query  p95  ${p(0.95).toFixed(1)} ms`);
console.log(`query  p99  ${p(0.99).toFixed(1)} ms`);
console.log(`query  min  ${t[0].toFixed(1)} ms   max ${t[t.length-1].toFixed(1)} ms`);
console.log(`payload     ${gz}`);
