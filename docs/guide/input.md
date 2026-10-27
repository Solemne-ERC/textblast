# Input

The input is one file in [JSON Lines](https://jsonlines.org/) format (`.jsonl`, or gzipped `.jsonl.gz`): one JSON
object per line, one document per object. `examples/quickstart/letters.jsonl` is a small example.

```json
{"doc_id": "letter-01", "text": "To the reeve and the good people of Ashford, greeting. …", "date": 1402, "series": "priory",
 "locs": [{"loc": "letter-01:p1", "start": 0, "length": 171}, {"loc": "letter-01:p2", "start": 173, "length": 389}, …]}
```

| field | required | |
|---|---|---|
| `doc_id` | yes | a non-empty string, unique in the file |
| `text` | yes | the text as transcribed; nothing needs to be cleaned beforehand |
| `locs` | no | units such as pages, folios or chapters: `[{loc, start, length}]` (name: `ingest.locs_field`) |
| `exclusions` | no | classified spans: `[{start, length, mode, reason}]` (name: `ingest.exclusions_field`) |
| anything else | no | metadata, carried into `documents` and the exports (e.g. `date`, `series`, `author`) |

Documents keep their order: the first line is document 0 (`doc_idx` in the tables). Offsets (`start`, `length`) are
counted in Unicode code points of `text`, starting at 0, exactly as Python's `text[start:start + length]`.
Metadata values are best kept to numbers and strings, so that filters can compare them (`date`, `series`).

## Units

Units are passages a reader cites by name: pages, folios, columns, chapters, letters in a register. They do not
change the search. They are used to:
- tell which units each passage overlaps (`tables/unit_overlaps`, and `locs1`/`locs2` in `pairs_flat`);
- optionally cut hits at unit boundaries (`units.clip: true`), when a hit across a page break should be two hits.

Units may leave gaps (e.g. between paragraphs) and may overlap. Their names (`loc`) are free text.

## Classified spans

Parts of a text that are not part of the work, or that should not produce matches, can be marked in the input:

| mode | effect |
|---|---|
| `delete` | the span is removed before the search, as if it were a space (e.g. page headers, editorial notes) |
| `mask` | the letters stay, but no match may *start* inside the span; a match may extend through it (e.g. a known formula you do not want reported on its own, but that should not break a longer match) |
| `barrier` | the span becomes a wall that no match can cross (e.g. a lacuna: text on either side is not continuous) |

`reason` is free text, counted in `logs/warnings/exclusions_by_reason.tsv` and kept in `corpus/exclusions.parquet`.
Spans of different modes may not overlap.

Spans can also come from a separate file (`ingest.exclusions_file`), one JSON object per line with `doc_id`, `start`,
`end` (exclusive), `mode` and `reason`. This keeps the corpus file unchanged when the spans come from another tool.

Two other sources of spans need no input at all:
- **declared gaps** (`normalize.declared_gaps`, on by default): `[...]`, `[lacuna]`, `[illegible]`, `[omitted]`,
  `...` and `…` become barriers, since the text is not continuous across them, and so do page scaffolding markers
  (`=== PAGE 12 ===`, `[CANVAS …]`, `[COLUMN …]`, `[REGION …]`, `[NO TEXT DETECTED]`);
- **pattern rules** (`normalize.pattern_rules`): regular expressions that delete, mask or wall off spans, optionally
  only in documents whose metadata match `where`. See [Configuration](configuration.md#normalize).

## Size

For scale: a corpus of 672 documents with 39 million letters prepares in a few seconds and searches in about three
minutes on a 14-core laptop, or two and a half minutes on a 64-thread server. See [Running](running.md#workers-and-threads).
