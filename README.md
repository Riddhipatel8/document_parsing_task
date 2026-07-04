# Document Extraction Pipeline

Turns a **lease PDF + a document type** (keys/descriptions + JSON Schema) into
**structured JSON that conforms to the schema** — designed to be cheaper and more
consistent than a per-key baseline, while staying accurate.

---

## Setup & run

```bash
# 1. install
python3 -m pip install -r requirements.txt

# 2. configure your key
cp .env.example .env
# edit .env and set GEMINI_API_KEY=...   (LiteLLM reads it automatically)

# 3. run
python extract.py --pdf samples/rent_notice.pdf     --doctype "Notice: Rent Change"
python extract.py --pdf samples/lease_commercial.pdf --doctype "Lease: Commercial" --out result.json

# consistency check: run N times, report field-level agreement
python extract.py --pdf samples/rent_notice.pdf --doctype "Notice: Rent Change" --repeat 3
```

Extracted JSON goes to stdout (or `--out`); cost, baseline comparison, validation
errors and consistency stats go to **stderr**, so stdout stays clean JSON.

The model is a single LiteLLM string in `pipeline/config.py`
(`EXTRACTION_MODEL`, default `gemini/gemini-2.5-flash`). Swap to
`gemini/gemini-2.5-pro`, `anthropic/claude-...`, `gpt-4o`, etc. with no code
change.

Tests run **without a key or network** (the LLM is mocked):

```bash
python -m pytest tests/ -q
```

---

## Design

> See [ARCHITECTURE.md](ARCHITECTURE.md) for the full architecture with Mermaid
> diagrams (pipeline flow, field→call division, the LLM call, and the repair loop).

```
PDF ─▶ text_extractor ─▶ (per-page image fallback for scanned pages)
        │
        ▼
document_type (parse fields: general vs individual, prompts, schemas)
        │
        ▼
planner  (general fields → 1 batched call; each individual field → 1 call)
        │
        ▼
llm_client (LiteLLM: stable cached doc block + per-field prompt, temp=0, seed)
        │
        ▼
validator (jsonschema validate → 1 repair call feeding back the error)
        │
        ▼
pipeline  (assemble → merged JSON) + cost accounting + baseline estimate
```

Each module has one job and a small interface, so pieces are independently
testable and the provider is swappable:

| Module | Responsibility |
|---|---|
| `text_extractor.py` | PDF → text; renders images only for pages with no text |
| `document_type.py` | Parse `document_types.json` into typed `Field`/`DocumentType` |
| `planner.py` | Group fields into the minimal set of calls |
| `llm_client.py` | LiteLLM wrapper: messages, JSON mode, temp/seed, usage |
| `validator.py` | Parse + schema-validate + build the repair message |
| `pipeline.py` | Orchestrate, assemble result, tally cost |
| `cost.py` | Real usage accounting + per-key-baseline estimate |

---

## How cost was reduced vs. the per-key baseline

The baseline rasterizes every page and makes **one LLM call per key**, re-sending
**all page images** each time. Three changes attack that:

1. **Text-first, not image-first.** Both samples are born-digital, so we send
   extracted *text* (≈10–50× cheaper than image tokens) and fall back to a page
   *image* only for pages with no extractable text. The 70-page lease is ~33k
   text tokens vs. ~18k *image* tokens **per key** in the baseline.
2. **Batch the cheap fields.** The five `general` scalar fields collapse into
   **one** call instead of five. Commercial lease: **8 calls → 4**; rent notice:
   **7 → 3**.
3. **Reuse the document via caching.** The document is a byte-identical leading
   block across all calls, so Gemini 2.5's **implicit prompt caching** discounts
   it on calls 2..N with no provider-specific code.

`cost.py` prints **real** measured tokens/$ per run plus an estimate of what the
baseline would have cost on the same document. Illustrative estimate from a dry
run (input tokens, dominant term):

| Document | Baseline (per-key, images) | This pipeline | Calls |
|---|---|---|---|
| Commercial lease (70 pp) | ~144k img-tokens | ~34k text-tokens × reuse¹ | 8 → 4 |
| Rent notice (1 pp) | ~1.8k img-tokens | ~0.1k text-tokens × reuse¹ | 7 → 3 |

¹ With implicit caching, the shared doc on calls 2..N is discounted further.
Numbers are estimates until you run with a key, at which point the tool reports
actual usage. (See "Tradeoffs" for the honest caveat on the heavy per-field
prompts.)

---

## How output was made consistent (and how to verify it)

Levers, in order of impact:

- **`temperature=0` + fixed `seed`** (both honored by Gemini).
- **Schema-constrained output**: JSON mode + **local `jsonschema` validation**
  against the field's real schema — enforcing `enum`, `required` and
  `additionalProperties:false` that native structured-output modes often ignore.
- **Deterministic assembly**: fields planned and emitted in a fixed, sorted order;
  identical prompt bytes every run.
- **Repair loop**: on validation failure, one corrective call feeds the exact
  error back, so we converge to conforming output instead of retrying blindly.

**Verify it:** `--repeat N` runs the same input N times and reports per-field and
overall exact-match agreement:

```bash
python extract.py --pdf samples/rent_notice.pdf --doctype "Notice: Rent Change" --repeat 5
# -> consistency: { per_field: {...}, overall: 0.98, runs: 5 }
```

---

## Tradeoffs & what I'd do next

- **Honest cost caveat.** The biggest single cost driver on the commercial lease
  isn't the document — it's the **per-field prompts themselves** (the
  `consideration` description is ~87k chars ≈ 21k tokens). The baseline pays that
  too, so we still win via text-vs-image + batching + caching, but prompt
  compression / prompt caching of the *instructions* is the next lever.
- **Text-first assumes mostly-digital PDFs.** A fully scanned document falls back
  to image tokens for every page, eroding the savings. Fine for these samples.
- **Local validation over native structured output.** More robust and
  provider-agnostic, at the cost of a possible repair round-trip. A provider that
  supports full JSON Schema strictly could drop the repair path.
- **No retrieval/chunking.** Gemini's large context fits the whole 70-page lease,
  which is simpler and more accurate. For documents beyond the window I'd add
  section-routing (feed only relevant sections per field).
- **Next with more time:** model routing (flash for general, pro for the heavy
  fields), explicit `CachedContent` for a guaranteed discount, self-consistency
  voting on flaky fields, and a golden-file accuracy harness.
```
