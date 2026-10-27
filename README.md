# textblast

Text reuse detection for historical document collections with a patched NCBI BLAST engine and a data-agnostic
Python pipeline. Pre-release: the first release, 0.9.0, is in preparation; installation, a user guide and citation
details follow with it.

```
pip install -e '.[dev]'
```

## Engine

```
textblast engine build          # BLAST+ 2.17.0 with the text-reuse patches, into ~/.cache/textblast
textblast engine check          # verify the patched engine
textblast engine info           # version, fingerprint, location, last check
```

## Corpus store

```
textblast prepare --config configs/example.yaml      # ingest → normalize → encode → <run_dir>/corpus
textblast alphabet suggest --input corpus.jsonl --profile latin-broad
```

Input is JSONL (or JSONL.gz), one document per line: `doc_id`, `text`, optional metadata, optional units
(`[{loc, start, length}]`) and optional classified spans (`[{start, length, mode, reason}]`, mode `delete`, `mask`
or `barrier`). Offsets are 0-based, end-exclusive Unicode code points of the text as given. The store keeps, for every
encoded letter, the exact span of the input text it comes from (`textblast.mapping.CorpusStore`).

## BLAST runs and raw hits

```
textblast run --config configs/example.yaml           # prepare → plan → blast → merge (each skipped if up to date)

textblast plan --config configs/example.yaml          # blocks, databases, jobs → <run_dir>/blast/plan.json, jobs.tsv
textblast blast --run-dir runs/example                # all jobs not done yet, longest first (rerun to resume)
textblast run-job --run-dir runs/example --job 17     # one job (cluster arrays)
textblast merge --run-dir runs/example                # → <run_dir>/tables/hsps.parquet
textblast inspect --run-dir runs/example              # status, also while a run is in progress
```

Every pair of documents is searched exactly once, and a document is never searched against itself (so repetition
inside one document is not reported). BLAST settings live in the `blast:` section of the configuration. The presets
are `textblast` (the default: `comp_based_stats 0`, `evalue 1e-3`), and `textreuse-blast` and `comhis` for parity
with those tools; `--no-btop` drops the alignment transcript. The patched matrix and the gap costs 3/11 are fixed. Run `textblast engine check` once before cluster jobs: `run-job` refuses an unchecked engine.

`tables/hsps.parquet` has one row per BLAST alignment: documents `doc1 < doc2`, surface offsets
`start1, end1, start2, end2` (0-based, end-exclusive code points of the input text), encoded offsets, BLAST's scores
and e-value, `btop` (written from the query's side, given by `query_side`) and `hsp_id`.

## Pieces, pairs and exports

```
textblast post --run-dir runs/example   # clip, join → defragment → annotate → hit filter → cluster
textblast export --run-dir runs/example --format bundle,pairs_flat   # also passim-pairwise, etl, passim-clusters
```

`post` turns raw hits into the normalized tables in `tables/`:
- `pieces`: passages per document, merged across hits by a rule (`defrag.rule`: `etl`, ReceptionReader's rule, by
  default; `node_similarity`; `passim`, passim's span linking; or your own function). Annotations:
  - `masked_share`;
  - `doc_degree`, the number of documents the passage is linked to (formulae link to many);
  - `ref_overlap_<name>` for each reference set (`annotate.reference_sets`, e.g. `bible: "doc_id LIKE 'bible:%'"`):
    the share of the passage covered by hits with documents of the set;
- `piece_map`: each hit side → its raw piece → its piece (provenance);
- `pairs`: pairs of pieces with `num_orig_links` (the hits behind them) and their best scores;
- `unit_overlaps`: which units each piece overlaps;
- optionally `hsps_clipped` (`units.clip: true`): hits cut at the unit boundaries they cross, as an extra view;
- `join_map`: which raw hits continue each other. Two hits of a pair belong to one group when they are consecutive in
  both documents, in the same order, at most `join.gap` letters apart (500 by default) in both, and not a repeat:
  hits that overlap by more than half of the shorter one in either document (`join.max_overlap`) are one phrase
  matched twice and stay apart. Raw hits stay the unit (joining is off by default); the map shows a reader which hits
  belong together without merging them;
- optionally `hsps_joined` (`join.enabled: true`): the groups joined into one hit each, as a view beside the raw hits;
  `defrag.view: joined` builds the pieces from it;
- optionally `pairs_filtered` (`filters.hit_filter`);
- optionally `clusters` and `cluster_summary` (`cluster.enabled: true`; `cluster.method`: `chinese_whispers` by
  default, or `components`): pieces grouped through their pairs. Connected components tend to link much of a corpus into one cluster; Chinese Whispers does not.
  The largest clusters by document count are often short stock phrases: `cluster_summary` has the median piece
  length and the aligned letters inside each cluster (`total_length`), to rank borrowings above them.

Nothing is filtered by default. `filters.pair_filter` / `filters.series_field` (document metadata, passim's
`field`/`field2` convention) remove pairs and let BLAST skip work; `filters.hit_filter` never removes anything from the
tables and only selects what goes into exports. The hit filter sees the piece annotations of both sides and the
pieces' lengths in letters (`piece_letters1`/`2`). Named filters can be used in it:
- `@no_formulae`: no side with `doc_degree` ≥ 30 and under 60 letters. On the corpus this removes 45% of the pairs
  for 0–3.4 points of recall; an expert judged 19 of 20 such pieces formulae;
- `@no_bible_mediated`: not both sides ≥ 80% covered by hits with Bible documents (needs the reference set `bible`);
- combined with anything else, e.g. `@no_formulae AND length >= 50`.

The core tables are the native format. `export` writes a portable `bundle` (tables, texts, manifest) or converts them:
`pairs_flat` (Parquet, CSV or JSONL), `passim-pairwise` (passim's `align.json` records), `etl` (the HPC-HD ETL /
ReceptionReader input) and `passim-clusters` (passim's cluster records, with `pboiler` from `export.boiler_field`),
with excerpts (`--excerpts none|surface`) and code-point or UTF-16 offsets.

```python
from textblast import open_run
run = open_run("runs/example")          # or a bundle
pairs = run.pairs.collect()            # Polars LazyFrames: documents, hsps, joined, pieces, pairs, clusters, …
run.excerpt(doc, start, end)           # surface text; run.to_utf16(doc, offsets) for JavaScript
```

## Evaluating against expected pairs

```
textblast evaluate --run-dir runs/example --expected expected.parquet [--alternatives alternatives.parquet] \
    [--level hits|clipped|joined|pairs]
```

`evaluate` measures recall against pairs you know to be reuse, e.g. references recorded by editors. Expected pairs
name documents by `doc_id` and give spans in code points of the input text, so one file serves every run of the same
input.

**Expected pairs** (Parquet, one row per pair):

| column | |
|---|---|
| `ref_id` | unique id |
| `kind`, `relation` | free text, e.g. `bible_quote` and `direct` or `cf`; results are tabulated by both |
| `doc_id1`, `start1`, `end1` | the annotated passage |
| `doc_id2`, `start2`, `end2` | the target |
| `similarity` | optional, 0–1: how much text the two sides share; recall is also tabulated by band |
| `target_group` | optional `doc_id` prefix, e.g. `bible:`: a hit on another document of the group counts as *elsewhere* |
| `exact1`, `exact2` | optional, default true: that side's boundaries are the reused text's own (only those get *overshoot*) |

Any other columns are ignored. **Alternative targets** (optional, Parquet): `ref_id`, `doc_id2`, `start2`, `end2`,
`match` (a label such as `near` or `parallel`).

Each expected pair gets the best of: **exact** (a hit overlaps the passage on one side and the target on the other,
in either order), **alternative** (the same against an alternative target), **elsewhere** (a hit overlaps the passage
and a document of the target group), or **missed**. Results also give the share of the passage the hits cover and, on
exact sides, the **overshoot** of the main hit: how many characters it runs past the passage at each end.

Output in `<run_dir>/evaluate/<expected>-<level>/` (or `--out`): `results.parquet` (one row per expected pair) and
`summary.json` (recall per `kind` × `relation`, per length band and per similarity band; overshoot per kind).

## Current limitations

- **Latin script only.** All profiles reduce text to the Latin letters a–z. Letters of other scripts are removed and
  listed in `logs/warnings/unsupported_chars.tsv`.
  - Support for non-Roman scripts, with particular attention to Cyrillic (romanization profiles), is planned for a
    version after the first release series.

The patched NCBI BLAST+ is public domain (US Government work); see NCBI's notices in the source distribution.

## Acknowledgements

Funded by the European Union (ERC, SOLEMNE, 101087979). Views and opinions expressed are however those of the
author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive
Agency. Neither the European Union nor the granting authority can be held responsible for them.

Developed by Gleb Schmidt (Radboud University, Faculty of Arts, Humanities Lab) with the assistance of Claude
(Anthropic).
