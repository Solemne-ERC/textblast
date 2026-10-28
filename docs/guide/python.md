# Python API and evaluation

## Reading a run

```python
import polars as pl
from textblast import open_run

run = open_run("runs/quickstart")        # a run directory, or a bundle made by `export --format bundle`
docs = run.documents.collect()           # every table is a Polars LazyFrame: nothing is read until collect()
pairs = run.pairs.collect()

# the ten longest borrowings, with their text
sides = run.pieces.select(pl.col("piece_id").alias("piece1"), pl.col("doc").alias("d1"),
                          pl.col("start").alias("s1"), pl.col("end").alias("e1"))
best = run.pairs.join(sides, on="piece1").sort("total_length", descending=True).head(10).collect()
for row in best.iter_rows(named=True):
    print(row["total_length"], run.excerpt(row["d1"], row["s1"], row["e1"])[:80])
```

| member | |
|---|---|
| `documents`, `hsps`, `pieces`, `piece_map`, `pairs`, `units`, `unit_overlaps`, `join_map` | the tables, as LazyFrames |
| `hsps_clipped`, `joined`, `pairs_filtered`, `clusters`, `cluster_summary` | the optional tables; reading one that was not made raises `RunNotAvailable`, with the setting that makes it |
| `table(name)` | any table by name |
| `manifest` | the manifest, as a dict |
| `texts()` | the input texts, as a list indexed by `doc_idx` |
| `excerpt(doc, start, end)` | the text of a span; with arrays, a list of texts |
| `to_utf16(doc, offsets)` | code-point offsets → UTF-16 offsets, for JavaScript |
| `to_surface(doc, enc_start, enc_end)` | encoded ranges → spans of the input text (run directories only) |

Files are the interface: the API only reads them, so a notebook can open a run while another process works on it,
and other tools can read the Parquet files directly. Each table's schema version is in the manifest (`schemas`).

## Evaluating against known pairs

```
textblast evaluate --run-dir runs/mycorpus --expected expected.parquet [--alternatives alternatives.parquet] \
    [--level hits|clipped|joined|pairs] [--out DIR]
```

`evaluate` measures recall against pairs you know to be reuse, such as references recorded by editors. Expected pairs
name documents by `doc_id` and give spans in code points of the input text, so one file serves every run of the same
input.

**Expected pairs** (Parquet, one row per pair):

| column | |
|---|---|
| `ref_id` | a unique id |
| `kind`, `relation` | free text, e.g. `bible_quote` and `direct` or `cf`; results are tabulated by both |
| `doc_id1`, `start1`, `end1` | the annotated passage |
| `doc_id2`, `start2`, `end2` | the target |
| `similarity` | optional, 0–1: how much text the two sides share; recall is also tabulated by band |
| `target_group` | optional `doc_id` prefix, e.g. `bible:`: a hit on another document of the group counts as *elsewhere* |
| `exact1`, `exact2` | optional, default true: whether that side's boundaries are those of the reused text (only those get *overshoot*) |

Other columns are ignored. **Alternative targets** (optional, Parquet): `ref_id`, `doc_id2`, `start2`, `end2`,
`match` (a label such as `near` or `parallel`).

Each expected pair gets the best of:
- **exact:** a hit overlaps the passage on one side and the target on the other, in either order;
- **alternative:** the same against an alternative target;
- **elsewhere:** a hit overlaps the passage and some document of the target group;
- **missed.**

The results also give the share of the passage the hits cover and, on exact sides, the **overshoot** of the main hit:
how many characters it runs past the passage at each end. `--level` chooses the hits: raw (`hits`), `clipped`,
`joined`, or the `pairs` of pieces.

Output in `<run_dir>/evaluate/<expected>-<level>/` (or `--out`): `results.parquet` (one row per expected pair) and
`summary.json` (recall per `kind` × `relation`, per length band and per similarity band; overshoot per kind).
`--length-bands 50,150,500` and `--similarity-bands 0.5,0.8` set the bands.
