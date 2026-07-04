#!/usr/bin/env python3
"""CLI entrypoint.

Examples:
  python extract.py --pdf samples/rent_notice.pdf --doctype "Notice: Rent Change"
  python extract.py --pdf samples/lease_commercial.pdf --doctype "Lease: Commercial" \
      --out result.json
  python extract.py --pdf samples/rent_notice.pdf --doctype "Notice: Rent Change" \
      --repeat 3        # consistency check across repeated runs
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pipeline import extract, get_document_type

DEFAULT_TYPES = Path(__file__).parent / "samples" / "document_types.json"

logger = logging.getLogger(__name__)


def _field_agreement(runs: list[dict]) -> dict:
    """Fraction of runs whose value for each field matches run 0 (canonical JSON)."""
    if not runs:
        return {}
    keys = sorted({k for r in runs for k in r})
    agreement = {}
    for k in keys:
        base = json.dumps(runs[0].get(k), sort_keys=True)
        matches = sum(
            1 for r in runs if json.dumps(r.get(k), sort_keys=True) == base
        )
        agreement[k] = round(matches / len(runs), 3)
    overall = round(sum(agreement.values()) / len(agreement), 3) if agreement else 0.0
    return {"per_field": agreement, "overall": overall, "runs": len(runs)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Document extraction pipeline")
    ap.add_argument("--pdf", required=True, help="Path to the input PDF")
    ap.add_argument("--doctype", required=True, help="Document type name")
    ap.add_argument("--types", default=str(DEFAULT_TYPES),
                    help="Path to document_types.json")
    ap.add_argument("--out", help="Write extracted JSON here (default: stdout)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="Run N times and report field-level consistency")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Enable DEBUG-level logging")
    args = ap.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    logger.info(
        "Starting extraction: pdf=%s doctype=%r repeat=%d",
        args.pdf, args.doctype, args.repeat,
    )
    logger.debug("Types file: %s", args.types)

    doc_type = get_document_type(args.types, args.doctype)
    logger.info(
        "Document type loaded: %r (%d fields)", doc_type.name, len(doc_type.fields)
    )

    runs = []
    last = None
    for i in range(args.repeat):
        logger.info("Run %d/%d", i + 1, args.repeat)
        last = extract(args.pdf, doc_type)
        runs.append(last.data)
        if args.repeat > 1:
            print(f"  run {i + 1}/{args.repeat} done "
                  f"({last.usage['calls']} calls, "
                  f"{last.usage['prompt_tokens']} prompt tokens)", file=sys.stderr)

    logger.info(
        "All runs complete. Total cost: $%.6f over %d call(s)",
        last.usage["total_cost_usd"], last.usage["calls"],
    )

    if last.errors:
        logger.warning("%d validation error(s): %s", len(last.errors), last.errors)

    result = last.data
    out_text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(out_text)
        logger.info("Output written to %s", args.out)
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(out_text)

    # Diagnostics go to stderr so stdout stays clean JSON.
    print("\n--- cost ---", file=sys.stderr)
    print(json.dumps(last.usage, indent=2), file=sys.stderr)
    print("\n--- baseline comparison ---", file=sys.stderr)
    print(json.dumps(last.baseline, indent=2), file=sys.stderr)
    if last.errors:
        print("\n--- validation errors ---", file=sys.stderr)
        print("\n".join(last.errors), file=sys.stderr)
    if args.repeat > 1:
        print("\n--- consistency ---", file=sys.stderr)
        print(json.dumps(_field_agreement(runs), indent=2), file=sys.stderr)

    return 1 if last.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
