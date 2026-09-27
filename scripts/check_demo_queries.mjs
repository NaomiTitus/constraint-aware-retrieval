// Run the PAGE'S OWN javascript against the shipped index, so what is checked is
// what ships. Not a relevance benchmark — no judgments — but a smoke test with a
// STATED expectation per query, so a regression becomes visible instead of a vibe.
import {readFileSync} from "fs";
const html=readFileSync("docs/index.html","utf8");
const js=html.slice(html.indexOf("<script>")+8,html.lastIndexOf("</script>"));
const el=()=>({innerHTML:"",textContent:"",value:"",hidden:true,disabled:false,
  children:[],addEventListener(){},dataset:{}});
globalThis.document={getElementById:el,querySelectorAll:()=>[]};
globalThis.fetch=async()=>({ok:true,json:async()=>({})});
const m=new Function(js.replace(/\(async\(\)=>\{[\s\S]*$/,"")+`
 ;return {parse,score,normPlace,set:(a,g)=>{ADS=a;GAZ=g.labels;TOKENS=g.tokens;
  TITLES=g.titles||{};NDOC=ADS.length;
  GAZKEYS=Object.keys(GAZ).sort((x,y)=>y.length-x.length);
  for(const ad of ADS){const seen=new Set(norm(ad.t+" "+ad.o+" "+ad.e+" "
    +(ad.k||[]).join(" ")+" "+(ad.g||[]).join(" ")).split(" "));
    for(const w of seen) if(w) DF[w]=(DF[w]||0)+1;}
  for(const ad of ADS)for(const [q,c] of (ad.L||[])){const nm=normPlace(q),nc=normPlace(c);
    if(nc)COUNTIES.add(nc); if(nm&&!(nm in MUNI))MUNI[nm]=nc||"";}}};`)();
const idx=JSON.parse(readFileSync("docs/data/index.json","utf8"));
m.set(idx.ads, JSON.parse(readFileSync("docs/data/occupations.json","utf8")));
const SPEC=JSON.parse(readFileSync("eval/demo_queries.json","utf8")).queries;
const K=5; let catHits=0, catTot=0, langFail=[], placeFail=[], parseFail=[];
for(const s of SPEC){
  const S=m.parse(s.q), r=m.score(S,0.7).slice(0,K);
  const inCat=r.filter(x=>x.ad.c===s.cat).length;
  catHits+=inCat; catTot+=K;
  if((S.norwegian||null)!==(s.lang||null)) langFail.push([s.q,S.norwegian,s.lang]);
  if(s.place && (!S.place || S.place.phrase.toLowerCase()!==s.place.toLowerCase()))
    placeFail.push([s.q,S.place&&S.place.phrase,s.place]);
  if(!S.styrk.length) parseFail.push(s.q);
  const blocked=r.filter(x=>x.sev>=.5).length;
  const flag = inCat>=3 ? "ok " : inCat>=2 ? "hm " : "BAD";
  console.log(`\n[${flag}] ${inCat}/${K} in "${s.cat}"  lang=${S.norwegian}`
    +`${s.lang==="none"?`  blocked ${blocked}/5`:""}  ${s.q.slice(0,52)}`);
  console.log(`        occ: ${S.evidence.occupation||"—"}`);
  for(const x of r) console.log(`        ${x.final.toFixed(2)} ${(x.ad.c||"?").slice(0,20).padEnd(22)}`
    +`${(x.ad.o||"").slice(0,24).padEnd(26)}${(x.ad.t||"").slice(0,34)}`);
}
console.log(`\n${"=".repeat(70)}`);
console.log(`industry match in top-${K}: ${catHits}/${catTot} = ${(catHits/catTot*100).toFixed(0)}%`);
console.log(`language parse failures: ${langFail.length}`);
langFail.forEach(([q,got,want])=>console.log(`   got ${got} want ${want}: ${q.slice(0,50)}`));
console.log(`place parse failures: ${placeFail.length}`);
placeFail.forEach(([q,got,want])=>console.log(`   got ${got} want ${want}: ${q.slice(0,50)}`));
console.log(`occupation unresolved: ${parseFail.length}`);
parseFail.forEach(q=>console.log(`   ${q.slice(0,56)}`));
