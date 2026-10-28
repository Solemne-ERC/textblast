# Outputs

## The run folder

```
runs/quickstart/
  config.yaml          the configuration used, with every default filled in
  manifest.json        what each stage did, with which settings, versions and engine
  corpus/              the corpus store (prepare)
  blast/               plan.json, jobs.tsv, the databases (db/) and each job's result (jobs/)
  tables/              the results: Parquet tables (merge, post)
  exports/             converted outputs (export)
  logs/                run.log, events.jsonl, warnings/, jobs/
```

`corpus/`, `tables/` and `manifest.json` are what readers use.

## Ids and offsets

- `doc_idx` (`doc`, `doc1`, `doc2` in the tables) is a document's position in the input, from 0; `documents` maps it
  to `doc_id`.
- In every pair and hit, `doc1 < doc2`.
- `start`, `end` (and `start1`, `end1`, `start2`, `end2`) are 0-based, end-exclusive code-point offsets of the input
  text: `text[start:end]` is the passage.
- `enc_start`, `enc_end` are the same range in the encoded sequence (letters only), used for joining and for measuring
  in letters.

## The corpus store: `corpus/`

| table | one row per | columns |
|---|---|---|
| `documents.parquet` | document | `doc_idx`, `doc_id`, `n_chars` (characters), `n_letters` (after normalization), `n_encoded`, `n_masked`, `n_barriers`, `deleted_share`, `empty` (no letters left: never searched), then the input's metadata fields |
| `texts.parquet` | document | `doc_idx`, `text` (the input text, unchanged) |
| `units.parquet` | unit | `doc_idx`, `loc`, `start`, `length` |
| `exclusions.parquet` | classified span | `doc_idx`, `start`, `end`, `mode`, `reason`, `source` (`field`, `file`, `pattern` or `gap`) |

`encoded.bin`, `encoded_index.npy`, `map_start.npy` and `map_end.npy` hold the encoded letters and, for each, the
span of the input text it came from.

## Hits: `tables/hsps.parquet` (merge)

One row per alignment BLAST found ("high-scoring segment pair").

| column | |
|---|---|
| `hsp_id` | 0, 1, 2, … |
| `doc1`, `doc2` | the documents (`doc1 < doc2`) |
| `start1`, `end1`, `start2`, `end2` | the passage in each document (code points) |
| `enc_start1`, `enc_end1`, `enc_start2`, `enc_end2` | the same in encoded letters |
| `length` | alignment columns, gaps included |
| `nident`, `mismatch`, `gapopen`, `gaps` | identical letters, mismatches, gap openings, gap columns |
| `pident`, `ppos` | percent identical, percent positive |
| `score`, `bitscore`, `evalue` | BLAST's raw score, bit score and e-value |
| `btop` | BLAST's alignment transcript: numbers are runs of identical letters, letter pairs are mismatches, `-` gaps |
| `query_side` | which side (1 or 2) was BLAST's query; `btop` is written from that side |
| `job` | the BLAST job that found it |

## Post-processing: `tables/` (post)

| table | one row per | columns |
|---|---|---|
| `join_map` | hit | `hsp_id`, `joined_id` (its group of neighbouring hits), `n_members` (hits in the group). Always written. |
| `hsps_joined` | group (`join.enabled`) | as `hsps`, over the whole group, plus `joined_id`, `n_members`, `max_gap` (the largest gap between members, in letters) and `first_hsp_id`; no `btop` |
| `hsps_clipped` | segment (`units.clip`) | `clip_id`, `hsp_id` (the hit it comes from), coordinates as in `hsps`, `loc1`, `loc2` (the units), `col_start`, `col_end` (the segment's columns in the alignment), `length`, `nident`, `mismatch`, `gapopen`, `gaps`, `pident`, and the parent hit's `parent_score`, `parent_bitscore`, `parent_evalue` |
| `pieces` | passage in one document | `piece_id`, `doc`, `start`, `end`, `enc_start`, `enc_end`, `n_raw_pieces` (hit sides merged into it), `n_hsps`, and the annotations `masked_share`, `doc_degree` (how many other documents it is linked to) and `ref_overlap_<set>` per reference set |
| `piece_map` | hit side | `hsp_id`, `side` (1 or 2), `raw_piece_id`, `piece_id`: from each hit to its piece |
| `pairs` | pair of pieces | `pair_id`, `piece1`, `piece2`, `doc1`, `doc2`, `num_orig_links` (hits behind the pair), `score` and `bitscore` (best), `evalue` (lowest), `mean_pident`, `total_length` (aligned columns, summed) |
| `pairs_filtered` | pair passing `filters.hit_filter` | as `pairs` |
| `unit_overlaps` | piece × unit it overlaps | `piece_id`, `doc`, `loc`, `overlap_chars`, `piece_share` (of the piece), `unit_share` (of the unit) |
| `clusters` | piece (`cluster.enabled`) | `piece_id`, `cluster_id` |
| `cluster_summary` | cluster | `cluster_id`, `n_pieces`, `n_documents`, `piece_length_median` (letters), `total_length` (aligned columns of the pairs inside) |

Pieces are built from the hits named by `defrag.view`: the raw hits by default, or the joined or clipped ones.

## The manifest

`manifest.json` has the package version, the schema version of every table, and one entry per stage: when it ran,
its fingerprint (a hash of its inputs and settings, which decides whether it must run again), the settings it used,
counts, timings and peak memory. `plan` records the engine (version, fingerprint, directory, and how it was found:
argument, environment, bundled or cache). `textblast inspect` summarizes it.

## Exports: `exports/`

`textblast export --format NAME[,NAME…]` (or `export.formats` for `run`):

| format | file | |
|---|---|---|
| `pairs_flat` | `pairs_flat.parquet` (or `.csv`, `.jsonl`) | one row per pair of pieces, ready for a spreadsheet or a notebook: `doc_id1`, `start1`, `end1`, `doc_id2`, `start2`, `end2`, the columns of `pairs`, each side's annotations and metadata (suffix `1` or `2`), `cluster1`/`cluster2` if clustered, `locs1`/`locs2` (units overlapped) and `text1`/`text2` (the passages) |
| `bundle` | `bundle/` | the tables, the documents and (with `export.texts`) the texts: a run without the corpus store and BLAST files, to share or archive; `open_run` reads it |
| `passim-pairwise` | `passim-pairwise.jsonl` | passim's pairwise alignment records (`align.json`): `uid`, `uid2`, `id`, `id2`, document fields, `begin`, `end`, `begin2`, `end2`, `s1`, `s2` (the aligned texts), `matches` |
| `passim-clusters` | `passim-clusters.jsonl` | passim's cluster records, one per passage: `cluster`, `size`, `uid`, `id`, `begin`, `end`, metadata, `locs`, `text`; with `export.boiler_field`, `pboiler` (needs `cluster.enabled`) |
| `etl` | `etl.jsonl` | the input format of the ETL / ReceptionReader tools: `text1_id`, `text1_text_start`, `text1_text_end`, `text1_text`, the same for `text2`, `align_length`, `positives_percent` |

With `filters.hit_filter` set, the exports contain only the selected pairs. `export.excerpts: none` leaves out the
texts; `export.offsets: utf16` converts offsets for JavaScript. A format of your own is a function that
takes a `textblast.export.Context` (the opened run, the options, the output folder, the hit filter and view) and
returns the path it wrote and the number of rows, registered as `export.converters: {name: "module:function"}`; the
built-in converters in `textblast.export` are examples.
