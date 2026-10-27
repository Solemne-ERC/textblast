# Recipes

Each recipe changes only `post` or `export` options unless it says otherwise, so trying it repeats no BLAST work.
The commands use the quickstart's configuration; replace it with yours.

## Filtering hits

Nothing is filtered by default: every hit stays in the tables. A **hit filter** selects which pairs go into
`tables/pairs_filtered` and into the exports, without deleting anything:

```
textblast post --config examples/quickstart/run.yaml --filters.hit_filter "length >= 50"
textblast export --config examples/quickstart/run.yaml --filters.hit_filter "length >= 50" --format pairs_flat
```

(or set `filters.hit_filter` in the configuration file, so that every command uses it.) The condition is SQL over
the hit's columns (`length`, `pident`, `evalue`, …), the annotations of the pieces on both sides (`doc_degree1`,
`doc_degree2`, `masked_share1`, `ref_overlap_bible1`, …), their lengths in letters (`piece_letters1`,
`piece_letters2`), and the documents' metadata (`date`, `date2`, `series`, `series2`, …).

On the quickstart, `length >= 50` keeps 15 of 23 hits: the dropped ones are short phrases such as "the autumn court"
or "To the reeve of Ashford".

## Formulae and stock phrases

Collections of letters, charters or sermons repeat formulae: greetings, closings, dating clauses, invocations. They
are real shared text, but they crowd out borrowings. Three tools help:
- **`doc_degree`** (on every piece): how many other documents the passage is linked to. Formulae link to many.
- **The named filter `@no_formulae`**: no side linked to 30 or more documents and shorter than 60 letters. On a
  corpus of 672 early modern Latin texts it removed 45% of the pairs for 0–3.4 points of recall, and 19 of 20 pieces
  it removes were judged formulae by an expert. Combine it with other conditions:
  `--filters.hit_filter "@no_formulae AND length >= 50"`.
- **`cluster_summary`** (with clustering): the largest clusters by document count are often formulae; their
  `piece_length_median` is short and their `total_length` small compared with clusters of real borrowings.

Known formulae can also be **masked** in the input or with a pattern rule (`action: mask`): BLAST then starts no match
inside them, but a longer borrowing that contains one is still found whole.

## Quotations from a reference text (e.g. the Bible)

Add the reference text to the corpus as ordinary documents with a recognizable `doc_id` (e.g. `bible:Gen.1`, one
document per chapter), and name them as a **reference set**:

```yaml
annotate:
  reference_sets:
    bible: "doc_id LIKE 'bible:%'"
```

Every piece then gets `ref_overlap_bible`: the share of it covered by hits with Bible documents. Two uses:
- find the quotations: pairs where one side is a Bible document;
- set aside reuse that only goes *through* the Bible (two texts quoting the same verse): the named filter
  `@no_bible_mediated` drops pairs whose both sides are at least 80% covered by Bible hits.

Recall against the quotations editors recorded is measured with `textblast evaluate`, with `target_group` set to
`bible:` in the expected pairs, so that a hit on a parallel passage elsewhere in the Bible counts as *elsewhere* rather than *missed*
(see [Evaluation](python.md#evaluating-against-known-pairs)).

## Comparing only some pairs of documents

These filters act before the search, so BLAST skips the work (they change `blast`, which then runs again):

```yaml
filters:
  series_field: series                    # skip pairs within one series (e.g. copies of one register)
  pair_filter: "abs(date - date2) <= 300"  # only documents at most 300 years apart
blast:
  group_blocks_by: series                 # so that whole jobs can be skipped
```

## Interrupted quotations: joining

A quotation broken by an insertion, an omission or a damaged passage comes out as two or more hits. `tables/join_map`
(always written) says which hits of a pair continue each other: consecutive in both documents, in the same order,
at most `join.gap` letters apart (500), and not one phrase matched twice (`join.max_overlap`). To use the joined hits:

```
textblast post --config examples/quickstart/run.yaml --join.enabled true --defrag.view joined
```

In the quickstart, the bridge order is quoted with a sentence inserted, and appears as two hits that joining makes
one. With a large `join.gap`, unrelated hits of the same two documents can be joined as well: check the `max_gap` of
the joined hits, or lower the gap.

## Page and chapter boundaries: clipping

A hit that runs across a page break or from the end of one chapter into the next is sometimes two borrowings. With
units in the input, `units.clip: true` writes `tables/hsps_clipped`, the hits cut at every unit boundary on either
side, and `defrag.view: clipped` builds the pieces from them. `tables/unit_overlaps` and `locs1`/`locs2` in
`pairs_flat` tell which units each passage touches, with or without clipping.

## Clusters

```
textblast post --config examples/quickstart/run.yaml --cluster.enabled true
textblast export --config examples/quickstart/run.yaml --cluster.enabled true --format passim-clusters
```

Clustering groups pieces linked through pairs: the copies of one passage across the collection. The default,
**Chinese Whispers**, splits large linked groups where the links are weak, with links weighted by aligned letters
(`cluster.weight: total_length`), so long borrowings hold together and short phrases do not glue unrelated passages
into one cluster. `cluster.method: components` (connected components) puts everything that is linked, however
weakly, into one cluster; on a large corpus that is often one giant cluster. `cluster.filtered: true` clusters only
the pairs that pass the hit filter.

## Page headers, folio numbers and other scaffolding

Pattern rules remove or wall off text that is not part of the work:

```yaml
normalize:
  pattern_rules:
    - {pattern: '\[fol\.?\s*\d+[rv]?\]', action: delete, reason: folio_marker}
    - {pattern: '\[p\.\s*\d+\]', action: barrier, reason: page_break}
```

Page markers of the form `=== PAGE 12 ===` and `[CANVAS …]`, `[COLUMN …]`, `[REGION …]` are already walled off
by `normalize.declared_gaps`. Use `barrier` where the text on either side is not continuous (a lacuna, the end of a page), `delete` where it is
(a marker inside a sentence). Each rule's count is in `logs/warnings/exclusions_by_reason.tsv`.

## A larger or smaller search

- Shorter quotations: `blast.evalue: 1e-2` finds more short hits (and many more formulae).
- Only long, sure matches: `blast.evalue: 1e-9`, or a hit filter on `length`.
- The settings of an earlier tool, for comparison: `blast.preset: textreuse-blast` or `comhis`, with
  `normalize.profile: compat-eng`.

## Workers and threads

See [Running](running.md#workers-and-threads): one thread per worker, as many workers as free cores.
