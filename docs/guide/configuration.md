# Configuration

A run is described by one YAML file. [`configs/example.yaml`](../../configs/example.yaml) lists every option at its
default, with a short comment; copy it and change `input` and `run_dir`. Options left out keep their defaults, so a
configuration can be as short as:

```yaml
input: examples/quickstart/letters.jsonl
run_dir: runs/quickstart
```

Every option can also be given on the command line, which overrides the file: `--normalize.profile latin`,
`--blast.workers 32`, `--join.enabled true`. `textblast run --help` lists them all. Each run writes the configuration
it used, with every default filled in, to `<run_dir>/config.yaml`.

Each stage is skipped when its inputs and its part of the configuration are unchanged, so changing a `post` option
and running again repeats only `post` and the exports, not the search.

## Top level

| option | default | |
|---|---|---|
| `input` | (required) | the corpus file, see [Input](input.md) |
| `run_dir` | (required) | where everything is written, see [Outputs](outputs.md) |
| `workers` | performance cores | processes for `prepare`, `post` and the exports. On Apple silicon the default counts only performance cores (efficiency cores are slow for this work); elsewhere all cores. |

## `ingest`

| option | default | |
|---|---|---|
| `locs_field` | `locs` | the field that holds units |
| `exclusions_field` | `exclusions` | the field that holds classified spans |
| `exclusions_file` | none | a separate JSONL file of spans (`doc_id`, `start`, `end`, `mode`, `reason`) |

## `normalize`

| option | default | |
|---|---|---|
| `profile` | `latin-broad` | how text becomes letters; see [Normalization](normalization.md#profiles) |
| `declared_gaps` | `true` | `[...]`, `[lacuna]`, `…` and the like become barriers |
| `high_deletion_share` | `0.12` | warn about documents that lose more than this share of their characters (spaces, punctuation and deleted `h` are not counted): often a sign of another script or of markup |
| `pattern_rules` | `[]` | regular expressions that delete, mask or wall off spans |
| `steps` | the profile's | replace the profile's steps, e.g. to add a character rule |

A pattern rule:

```yaml
normalize:
  pattern_rules:
    - pattern: '\[fol\.?\s*\d+[rv]?\]'   # a Python regular expression
      action: delete                    # delete | mask | barrier
      reason: folio_marker              # counted in logs/warnings/exclusions_by_reason.tsv
      where: {source_type: transcription}   # optional: only documents with this metadata
```

Spans given in the input win over pattern rules, and pattern rules over declared gaps, where they overlap.

## `encode`

| option | default | |
|---|---|---|
| `barrier_length` | `100` | how many `X` letters stand for one barrier. 100 holds under every supported BLAST setting; the run checks this before it starts and refuses settings that would let a match cross a barrier. |

## `blast`

| option | default | |
|---|---|---|
| `preset` | `textblast` | a named set of the five values below; see [BLAST settings](normalization.md#blast-settings) |
| `evalue` | `1e-3` | the highest e-value reported. Lower it (e.g. `1e-9`) to keep only longer, surer matches. |
| `word_size` | `6` | letters in a seed word |
| `threshold` | `400` | seed score; with the patched matrix, 400 means only exact seed words |
| `window_size` | `40` | two seeds within this many letters start an extension; `0` starts one from a single seed |
| `comp_based_stats` | `0` | `0` scores with the patched matrix only. `2` re-aligns with BLOSUM62 statistics, as textreuse-blast did; it finds fewer, more tightly bounded hits and makes the results depend on the letter table. |
| `xdrop_ungap`, `xdrop_gap`, `xdrop_gap_final` | BLAST's | how far an extension may run through a bad stretch |
| `searchsp` | none | a fixed search space, so that e-values do not depend on document length |
| `max_target_seqs` | every sequence | BLAST's limit on documents reported per query |
| `workers` | performance cores | parallel BLAST jobs; see [Running](running.md#workers-and-threads) |
| `threads_per_worker` | `1` | threads per BLAST job |
| `block_letters` | about 10 jobs per worker | the size of a block of documents. Smaller blocks spread the work better over many workers, larger ones repeat less work per job. |
| `group_blocks_by` | none | order documents by a metadata field before blocking, so that a pair filter on that field can skip whole jobs |
| `btop` | `true` | keep BLAST's alignment transcript (needed for alignment strings and for `units.clip`) |
| `keep_raw` | `false` | keep BLAST's own tabular output next to each job |

The matrix (passed to BLAST under the name `BLOSUM62`) and the gap costs 3/11 are fixed by the patches.

## `engine`

| option | default | |
|---|---|---|
| `version` | `2.17.0` | `2.13.0` is the reference engine, for comparisons with earlier tools |
| `blast_bin` | none | a directory with `blastp`, `makeblastdb` and `blastdbcmd` to use instead |
| `force` | `false` | run even if the engine failed its last `engine check` |

## `filters`

Nothing is filtered by default. Two filters delete pairs of documents before the search, one selects hits afterwards.

| option | default | |
|---|---|---|
| `series_field` | none | a metadata field: pairs of documents with the same value are not compared (e.g. two copies of one edition) |
| `pair_filter` | none | a condition on the metadata of two documents, written with `field` for one and `field2` for the other, e.g. `abs(date - date2) <= 300`. Pairs that fail it are not compared. |
| `hit_filter` | none | a condition on hits, e.g. `length >= 50`. It removes nothing from the tables; it selects what goes into `tables/pairs_filtered` and the exports. See [Recipes](recipes.md#filtering-hits). |

Both conditions are SQL expressions (Polars SQL).

## `units`

| option | default | |
|---|---|---|
| `clip` | `false` | also write `tables/hsps_clipped`: hits cut at every unit boundary they cross, on either side |

## `join`

`tables/join_map` always records which hits of a pair continue each other; `enabled` also builds the joined hits.

| option | default | |
|---|---|---|
| `enabled` | `false` | also write `tables/hsps_joined`, one hit per group |
| `gap` | `500` | the largest gap, in letters, in both documents (about 600 characters) |
| `max_overlap` | `0.5` | no joint between hits that overlap by more than this share of the shorter one, in either document: those are usually one phrase matched twice. `null` turns the guard off. |
| `max_drift` | none | limit the difference between the two gaps, in letters |
| `min_gap_ratio` | none | the shorter gap must be at least this share of the longer one (guards against long insertions on one side) |

## `defrag`

Defragmentation turns hit sides into pieces: passages per document, merged across overlapping hits.

| option | default | |
|---|---|---|
| `rule` | `etl` | `etl` (the rule of the ETL/ReceptionReader tools), `node_similarity`, `node_similarity-compat`, `passim` (passim's span linking), or your own function as `module:function` |
| `node_similarity` | `0.9` | threshold of the `node_similarity` rules |
| `src_overlap`, `dst_overlap` | `0.9`, `0.5` | thresholds of the `passim` rule |
| `view` | `hsps` | which hits to defragment: `hsps` (raw), `clipped` (needs `units.clip`) or `joined` (needs `join.enabled`) |

## `annotate`

| option | default | |
|---|---|---|
| `reference_sets` | none | named sets of documents, as conditions on `documents`, e.g. `{bible: "doc_id LIKE 'bible:%'"}`. Each piece gets `ref_overlap_<name>`: the share of it covered by hits with documents of the set. |

## `cluster`

| option | default | |
|---|---|---|
| `enabled` | `false` | write `tables/clusters` and `tables/cluster_summary` |
| `method` | `chinese_whispers` | `chinese_whispers`, `components` (connected components) or `module:function` |
| `weight` | `total_length` | Chinese Whispers' edge weight: `total_length`, `num_orig_links`, `score` or `none` |
| `iterations` | `50` | Chinese Whispers: the most rounds (it stops earlier when no label changes) |
| `batches` | `10` | Chinese Whispers: random batches per round |
| `seed` | `0` | Chinese Whispers: the random seed, for repeatable clusters |
| `filtered` | `false` | cluster only the pairs that pass the hit filter |

## `export`

| option | default | |
|---|---|---|
| `formats` | `[]` | formats that `run` writes at the end; `export --format` writes any of them on demand |
| `excerpts` | `surface` | include the text of each side (`surface`) or not (`none`) |
| `offsets` | `codepoints` | `utf16` for use in JavaScript |
| `fields` | all metadata | which metadata fields to carry |
| `file_format` | `parquet` | `pairs_flat`: `parquet`, `csv` or `jsonl` |
| `view` | `defrag.view` | which hits `etl` and `passim-pairwise` export: `hsps`, `clipped` or `joined` |
| `texts` | `true` | `bundle`: include the texts |
| `boiler_field` | none | `passim-clusters`: a metadata field; `pboiler` is the share of a cluster's links within one value of it |
| `out_dir` | `<run_dir>/exports` | where exports go |
| `converters` | none | your own formats: `{name: "module:function"}` |
