"""
Build rag/subset_200.json: a fixed 200-sample subset for judge stability runs.

Stratified sampling: each benchmark contributes proportionally to its
size (SIUO 167, MSSB 300, MMSB 200, SPAVLH 265 -> ~36/64/43/57 of 200).
Within each benchmark, samples are balanced on the rag_on_response
safe/unsafe label as far as the actual distribution allows.

Deterministic: sorted by (label, image path), then first n per half,
so re-running produces the same manifest.
"""
import json
import pathlib

RESULT = pathlib.Path(__file__).parent / "result"
MANIFEST = pathlib.Path(__file__).parent / "subset_200.json"

BENCH_SIZES = {"SIUO": 167, "MSSB": 300, "MMSB": 200, "SPAVLH": 265}
TOTAL = 200

# proportional allocation of 200, adjusted so it sums to exactly 200
total_n = sum(BENCH_SIZES.values())
alloc = {b: round(n / total_n * TOTAL) for b, n in BENCH_SIZES.items()}
diff = TOTAL - sum(alloc.values())
# fix rounding by adjusting the largest benchmark
alloc["MSSB"] += diff

manifest = {}
for prefix, n in alloc.items():
    path = RESULT / f"JUDGED_{prefix}_internvl3_5-8b.json"
    data = json.load(open(path, encoding="utf-8"))
    safe, unsafe = [], []
    for r in data:
        rag = r.get("rag_on_response") or {}
        if rag.get("is_safe") is True:
            safe.append(r["image"])
        elif rag.get("is_safe") is False:
            unsafe.append(r["image"])
    safe.sort()
    unsafe.sort()
    half = n // 2
    pick = safe[:half] + unsafe[: n - half]
    # if a class is short of half, backfill from the other class
    if len(pick) < n:
        pool = [x for x in safe + unsafe if x not in pick]
        pick += pool[: n - len(pick)]
    pick.sort()
    manifest[prefix] = {
        "count": len(pick),
        "rag_safe_in_subset": sum(
            1 for r in data if r["image"] in pick and (r.get("rag_on_response") or {}).get("is_safe") is True
        ),
        "rag_unsafe_in_subset": sum(
            1 for r in data if r["image"] in pick and (r.get("rag_on_response") or {}).get("is_safe") is False
        ),
        "images": pick,
    }
    print(f"{prefix}: {len(pick)} samples (RAG safe {manifest[prefix]['rag_safe_in_subset']}, unsafe {manifest[prefix]['rag_unsafe_in_subset']})")

json.dump(manifest, open(MANIFEST, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"total: {sum(m['count'] for m in manifest.values())} -> {MANIFEST}")
