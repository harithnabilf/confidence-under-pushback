"""Fetch a stratified random sample of MMLU-Pro (TIGER-Lab/MMLU-Pro, test split) via the HF datasets-server API."""
import json, random, requests, collections, sys, time

N_TOTAL = 12032
SEED = 20261115
rng = random.Random(SEED)
URL = "https://datasets-server.huggingface.co/rows"

rows = []
offsets = rng.sample(range(0, N_TOTAL - 30, 30), 40)
for off in offsets:
    for attempt in range(8):
        r = requests.get(URL, params=dict(dataset="TIGER-Lab/MMLU-Pro", config="default", split="test", offset=off, length=30), timeout=60)
        if r.status_code == 429:
            time.sleep(3 * (attempt + 1)); continue
        r.raise_for_status(); break
    time.sleep(0.5)
    rows += [x["row"] for x in r.json()["rows"]]
print("fetched", len(rows), file=sys.stderr)

def size(r):
    return len(r["question"]) + sum(len(o) for o in r["options"])

# short items only (token budget), at least 4 options
cands = [r for r in rows if size(r) < 650 and len(r["options"]) >= 4]
by_cat = collections.defaultdict(list)
for r in cands:
    by_cat[r["category"]].append(r)
for v in by_cat.values():
    rng.shuffle(v)
# round-robin across categories for balance
picked = []
while len(picked) < 200 and any(by_cat.values()):
    for c in sorted(by_cat):
        if by_cat[c]:
            picked.append(by_cat[c].pop())
rng.shuffle(picked)
with open("data/questions.jsonl", "w", encoding="utf-8") as f:
    for r in picked:
        f.write(json.dumps(dict(qid=r["question_id"], category=r["category"], question=r["question"],
                                options=r["options"], answer_index=r["answer_index"], src=r.get("src")), ensure_ascii=False) + "\n")
print("kept", len(picked), collections.Counter(r["category"] for r in picked), file=sys.stderr)
