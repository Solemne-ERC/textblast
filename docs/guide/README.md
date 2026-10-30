# textblast user guide

textblast finds passages that two documents share: quotations, borrowings, copied formulae, reused paragraphs. It is
built for historical collections, where the same text appears with different spellings, abbreviations, gaps and
transcription errors, and it works on any collection of texts in the Latin script.

1. [Input](input.md): the JSON Lines format, units, and spans to delete, mask or wall off.
2. [Configuration](configuration.md): every option, its default, and when to change it.
3. [Normalization and BLAST settings](normalization.md): profiles, letter tables, presets, and why the defaults are
   what they are.
4. [Running](running.md): the stages and commands, resuming, logs, computing clusters (Slurm), and choosing workers
   and threads.
5. [Outputs](outputs.md): the run folder, every table and column, offsets, the manifest, and the export formats.
6. [Python API and evaluation](python.md): `open_run`, excerpts, and recall against known pairs.
7. [Recipes](recipes.md): filtering formulae and Bible quotations, joining, clustering, and other common tasks.

## How it works

The method is not new: it is that of Vesanto et al. (2017) and their tool textreuse-blast, which textblast
re-implements. The README's [Background](../../README.md#background) says what comes from which tool.

1. **Prepare.** Each text is normalized (case, accents, spelling variants such as `u`/`v`, abbreviation marks) and
   reduced to a small alphabet of letters. Every letter is encoded as an amino-acid symbol, so that the text becomes a
   protein-like sequence. The corpus store keeps, for every encoded letter, the exact span of the input it came from.
2. **Search.** NCBI BLAST+ compares every pair of documents once. textblast uses BLAST+ 2.17.0 with textreuse-blast's two
   patches for text: a uniform scoring matrix (+9 for a match, −5 for a mismatch, the same for every letter) and gap costs to
   match. The documents are split into blocks, and each pair of blocks is one job, so the work spreads over all cores
   and an interrupted run resumes where it stopped.
3. **Merge.** The jobs' hits are mapped back to exact character offsets of the input and gathered into one table.
4. **Post-process.** Hits become *pieces* (passages per document) and *pairs* of pieces, with annotations such as how
   many documents a passage is linked to. Optionally, neighbouring hits are joined, hits are cut at page or chapter
   boundaries, and pieces are clustered. Nothing is deleted: filters only select what goes into exports.
5. **Export.** The tables are Parquet files; exports convert them into flat tables with excerpts, a portable bundle,
   or the formats of passim and of the ETL/ReceptionReader tools.

## Offsets

Every offset in every table and export is a 0-based, end-exclusive count of Unicode code points of the input text as
given, so `text[start:end]` in Python is the passage. JavaScript counts UTF-16 code units instead: use
`export.offsets: utf16` or `Run.to_utf16` (they differ only for characters outside the Basic Multilingual Plane).

## Limitations

- **Latin script only.** Every profile reduces text to the letters a–z. Letters of other scripts are removed and
  listed in `logs/warnings/unsupported_chars.tsv`. Profiles for other scripts are planned.
- **Repetition inside one document is not reported:** a document is never compared with itself.
- **Platforms:** wheels for macOS on Apple silicon and Linux x86_64. Other Unix systems install from source and build
  the engine (`textblast engine build`); Windows works through WSL.
