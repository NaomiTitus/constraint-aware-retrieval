import sys, json
sys.path.insert(0,'src')
from finn_smart_search.ingest import store
from finn_smart_search.esco import fetch
con = store.connect('data/ads.duckdb')
uris = json.load(open('data/esco_uris.json'))
print(fetch.run(con, uris, log=lambda m: print(m, flush=True)), flush=True)
