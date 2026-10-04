"""Read-only inspection of real SITREP PDFs: hash, page text, cross-report diff. No LLM."""
import hashlib, json, re, sys, fitz
from pathlib import Path
up = Path("/mnt/user-data/uploads"); files = sorted(up.glob("Sitrep_*.pdf"))
man, texts = [], {}
for f in files:
    b = f.read_bytes()
    with fitz.open(stream=b, filetype="pdf") as d:
        pages = {i: p.get_text() for i, p in enumerate(d, 1)}
    texts[f.name] = pages
    man.append({"filename": f.name, "sha256": hashlib.sha256(b).hexdigest(), "file_size": len(b),
                "pages": len(pages), "chars_per_page": {i: len(t.strip()) for i, t in pages.items()}})
print(json.dumps(man, indent=1))
json.dump(man, open("benchmark/metadata/inspection.json", "w"), indent=1)
# cross-report diff after masking dates / report numbers / bulletin numbers
def mask(t):
    t = re.sub(r"\d{2}-\d{2}-\d{4}", "<D>", t); t = re.sub(r"\d{2}-August-\d{4}", "<D>", t)
    t = re.sub(r"SITREP-\d+\)/2025/\d+", "<N>", t); t = re.sub(r"B-0\d+/25", "<B>", t)
    t = re.sub(r"August \d\s*\w*,? ?\d{4}|August\s*\d+\s*(?:rd|th)?", "<D>", t)
    return t
names = list(texts)
for a, b in zip(names, names[1:]):
    print(f"\n=== {a} vs {b}: pages differing after masking dates ===")
    for i in texts[a]:
        la = [l.strip() for l in mask(texts[a][i]).splitlines() if l.strip()]
        lb = [l.strip() for l in mask(texts[b].get(i, "")).splitlines() if l.strip()]
        if la != lb:
            sa, sb = set(la), set(lb)
            print(f"page {i}: only-in-first={sorted(sa-sb)[:12]} only-in-second={sorted(sb-sa)[:12]}")
        else:
            print(f"page {i}: IDENTICAL")
print("\n--- raw fitz text, page 5 (cumulative damages table) first 40 lines ---")
print("\n".join(texts[names[0]][5].splitlines()[:40]))
