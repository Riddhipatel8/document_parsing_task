"""Parse `document_types.json` into typed objects.

A *document type* describes what to extract from a class of documents. It is a
list of *fields*; each field is either:

* ``general``    - a simple scalar (name, address, date). Empty per-field schema.
                   These are cheap and share context, so we batch them into ONE call.
* ``individual`` - a complex structured extraction carrying its own large prompt
                   (the ``description``) and its own JSON Schema. One call each.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Field:
    name: str
    description: str
    field_type: str  # "general" | "individual"
    item_type: str  # "string" | "object" | ...
    schema: dict[str, Any] = dataclass_field(default_factory=dict)

    @property
    def is_individual(self) -> bool:
        return self.field_type == "individual"

    @property
    def has_schema(self) -> bool:
        return bool(self.schema)


@dataclass(frozen=True)
class DocumentType:
    name: str
    fields: list[Field]

    @property
    def general_fields(self) -> list[Field]:
        return [f for f in self.fields if not f.is_individual]

    @property
    def individual_fields(self) -> list[Field]:
        return [f for f in self.fields if f.is_individual]


def _parse_field(raw: dict[str, Any]) -> Field:
    f = Field(
        name=raw["name"],
        # description is the extraction prompt; some general fields omit it.
        description=(raw.get("description") or "").strip(),
        field_type=raw.get("field_type", "general"),
        item_type=raw.get("item_type", "string"),
        schema=raw.get("schema") or {},
    )
    logger.debug(
        "  Field: %-30s type=%-10s item_type=%-8s has_schema=%s",
        f.name, f.field_type, f.item_type, f.has_schema,
    )
    return f


def load_document_types(path: str | Path) -> dict[str, DocumentType]:
    """Load all document types from the JSON file, keyed by name."""
    path = Path(path)
    logger.info("Loading document types from: %s", path)
    data = json.loads(path.read_text())
    types: dict[str, DocumentType] = {}
    for raw in data:
        fields = [_parse_field(f) for f in raw.get("fields", [])]
        types[raw["name"]] = DocumentType(name=raw["name"], fields=fields)
        logger.debug(
            "  Loaded type %r: %d fields (%d general, %d individual)",
            raw["name"],
            len(fields),
            sum(1 for f in fields if f.field_type == "general"),
            sum(1 for f in fields if f.field_type == "individual"),
        )
    logger.info("Loaded %d document type(s): %s", len(types), sorted(types))
    return types


def get_document_type(path: str | Path, name: str) -> DocumentType:
    logger.debug("Requesting document type %r from %s", name, path)
    types = load_document_types(path)
    if name not in types:
        logger.error(
            "Document type %r not found. Available: %s", name, sorted(types)
        )
        raise KeyError(
            f"Document type {name!r} not found. Available: {sorted(types)}"
        )
    doc_type = types[name]
    logger.info(
        "Using document type %r: %d general + %d individual fields",
        doc_type.name,
        len(doc_type.general_fields),
        len(doc_type.individual_fields),
    )
    return doc_type
