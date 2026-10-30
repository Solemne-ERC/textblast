# Changelog

## Unreleased

## 0.9.1 (2026-10-30)

Credits, licences and metadata. Commands, configuration and outputs are the same as in 0.9.0.

- Package metadata: classifiers (Python 3.10–3.14, macOS and Linux) and keywords, for PyPI.
- README: badges for the DOI, the PyPI version and CI; the concept DOI in the citation section; a note that macOS's
  own Python 3.9 is too old.
- `RELEASING.md`: how a release is made. CI also runs weekly.
- Credit for the method: the README says plainly that the method is not new and that textblast is a newer and faster
  implementation of the method of Vesanto et al. (2017) and their tool textreuse-blast. A new Background section says
  what comes from which tool (the BLAST patches, the merge rules, the compatibility profiles and presets), with
  references. `CITATION.cff` and `.zenodo.json` list the same references. `licenses/` adds textreuse-blast's MIT
  licence, which applies to its letter tables, included in textblast, and lists what is taken from each tool.

## 0.9.0 (2026-10-28)

The first public release.

- **Engine:** NCBI BLAST+ 2.17.0 with two patches for text (a uniform +9/−5 match/mismatch matrix, and gap costs
  3/11 with statistics to match). Platform wheels for macOS arm64 (11.0 and later) and Linux x86_64
  (manylinux_2_28) carry `blastp`, `makeblastdb` and `blastdbcmd`; elsewhere, `textblast engine build` downloads
  NCBI's source, checks it, applies the patches and compiles it. `textblast engine check` tests the engine on
  synthetic sequences (matrix, gap costs, exact seeds, barriers, soft masking) and every run records which engine
  it used.
- **Input:** JSON Lines (optionally gzipped), one document per line, with optional metadata, units (pages, chapters)
  and classified spans to delete, mask or wall off. Offsets everywhere are 0-based, end-exclusive Unicode code points
  of the input text.
- **Normalization** for the Latin script (`latin-broad` by default, `latin`, and profiles compatible with earlier
  tools), with an exact map from every encoded letter back to the input.
- **Search:** every pair of documents once, in blocks sized for the number of workers; resumable; `plan`, `run-job`
  and `merge` for cluster job arrays.
- **Post-processing:** hits clipped at unit boundaries (optional), neighbouring hits joined (a map on every run, the
  joined view optional, with a guard against one phrase matched twice), passages per document (`etl`,
  `node_similarity` or `passim` rules), annotations (`doc_degree`, masked share, overlaps with reference sets such
  as the Bible), hit filters (including `@no_formulae` and `@no_bible_mediated`), and clustering (Chinese Whispers
  or connected components). Nothing is filtered by default.
- **Outputs:** Parquet tables, a portable bundle, and exports to `pairs_flat`, passim's pairwise and cluster
  records, and the ETL/ReceptionReader format; a Python API (`open_run`) with excerpts and UTF-16 offsets.
- **Evaluation:** recall against expected pairs (`textblast evaluate`), by kind, length and similarity.
- **Documentation:** a user guide (`docs/guide/`), an example configuration with every option, and a quickstart on
  a small invented corpus (`examples/quickstart/`).
