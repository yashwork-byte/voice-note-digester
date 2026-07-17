"""Search interface placeholder (D006).

Planned interface: a natural-language query is embedded once and run against
both vec tables; results are merged with entity-type labels so action items
and note content rank as genuinely different result kinds. Metadata filters
(sender, date range, confidence) narrow the SQL side before the vector match.
"""

from .schema import ActionItemRecord, NoteRecord


def search(query: str, top_k: int = 5) -> list[NoteRecord | ActionItemRecord]:
    raise NotImplementedError("Search not implemented yet — see decisions.md D006")


if __name__ == "__main__":
    raise SystemExit("Search not implemented yet — see decisions.md D006")
