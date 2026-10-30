# textblast

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23729155.svg)](https://doi.org/10.5281/zenodo.23729155)
[![PyPI](https://img.shields.io/pypi/v/textblast)](https://pypi.org/project/textblast/)
[![CI](https://github.com/Solemne-ERC/textblast/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Solemne-ERC/textblast/actions/workflows/ci.yml)

Text reuse detection for historical document collections. textblast finds the passages that documents share
(quotations, borrowings, copied formulae, reused paragraphs) with NCBI BLAST+, patched for text, and reports each
one at its exact place in your original texts, whatever spelling, abbreviations and transcription errors stood in the
way. It works on any collection of texts in the Latin script.

**The method is not new.** Finding text reuse by writing letters as amino-acid symbols and aligning the texts with
BLAST was introduced by Aleksi Vesanto and colleagues at the University of Turku (Vesanto et al. 2017), with their
tool textreuse-blast and a BLAST patched for text. textblast is a newer and faster implementation of their method:
[Background](#background) says what comes from where.

- **Input:** one JSON Lines file, one document per line, with optional metadata, page or chapter units, and spans to
  delete, mask or wall off.
- **Output:** Parquet tables of hits, passages and pairs, with exact character offsets; flat tables with excerpts;
  and the formats of passim and of the ETL/ReceptionReader tools.
- **Scale:** every pair of documents is compared, in parallel, resumably, on a laptop or a computing cluster. A corpus
  of 39 million letters takes about three minutes on a 14-core laptop.

## Installation

```
pip install textblast
```

Python 3.10 or newer. The wheels for **macOS on Apple silicon** (macOS 11 or newer) and **Linux x86_64** carry the
patched BLAST engine, so nothing else is needed. Check it with `textblast engine check`. (macOS's own `python3` is
3.9: with it, pip answers only that no matching distribution was found. Use a newer Python, e.g. from python.org,
Homebrew or pyenv.)

**Other platforms** (Linux on ARM, Intel Macs) install from source and build the engine once:

```
pip install textblast
textblast engine build        # downloads NCBI's source, checks and patches it, compiles it (5–45 minutes)
```

This needs a C/C++ compiler, `make`, and the development files of SQLite, zlib and bzip2 (on Debian or Ubuntu:
`apt install build-essential libsqlite3-dev zlib1g-dev libbz2-dev`; on macOS, the Xcode command line tools).
To use BLAST programs you built yourself, pass their directory with `--blast-bin`.

**Windows:** use the Linux version in [WSL](https://learn.microsoft.com/windows/wsl/install).

## Quickstart

The repository has a small example: eight invented letters that reuse each other's text
([`examples/quickstart`](examples/quickstart)).

```
git clone https://github.com/Solemne-ERC/textblast.git
cd textblast
textblast run --config examples/quickstart/run.yaml       # prepare → plan → blast → merge → post → export
textblast inspect --run-dir runs/quickstart
```

```
run directory runs/quickstart
prepare   8 documents · 2.8k letters · profile latin-broad · 0.26 s
plan      28 jobs · 8 blocks · engine 2.17.0 (5117e33dd929) · evalue 0.001 · cbs 0
blast     28/28 jobs done (100% of the work) · 0 failed · 23 hits · 19s of job time
merge     23 hits · 18 document pairs · 8.8 KB
post      35 raw pieces → 24 pieces · 23 pairs · rule etl on hsps
export    pairs_flat: 23 rows · 12.8 KB · runs/quickstart/exports/pairs_flat.parquet
```

(Times and the engine's fingerprint differ from machine to machine.) `runs/quickstart/exports/pairs_flat.parquet`
has one row per pair of passages, with both texts:

```python
import polars as pl

pairs = pl.read_parquet("runs/quickstart/exports/pairs_flat.parquet")
print(pairs.sort("total_length", descending=True).select("doc_id1", "doc_id2", "total_length", "text1").head(5))
```

The mill order of 1402 is found in its copy of 1431, despite the old spellings (`myller`, `bringeth`, `hath`), and in
the court record of 1440, which quotes part of it. The bridge order comes out as two hits, because a later copy
inserts a sentence; `--join.enabled true` makes them one. A closing formula links five letters.

For your own texts, copy [`configs/example.yaml`](configs/example.yaml), which lists every option at its default,
and set `input` and `run_dir`.

## Documentation

The [user guide](docs/guide/README.md):
[input](docs/guide/input.md) ·
[configuration](docs/guide/configuration.md) ·
[normalization and BLAST settings](docs/guide/normalization.md) ·
[running, resuming, clusters, workers](docs/guide/running.md) ·
[outputs](docs/guide/outputs.md) ·
[Python API and evaluation](docs/guide/python.md) ·
[recipes](docs/guide/recipes.md).
Changes are listed in the [changelog](CHANGELOG.md).

## Limitations

- **Latin script only.** Letters of other scripts are removed and listed in `logs/warnings/unsupported_chars.tsv`.
  Support for other scripts, with particular attention to Cyrillic, is planned for a later version.
- Repetition inside one document is not reported: a document is never compared with itself.

## Background

textblast does not introduce a new method. It re-implements an existing one, together with the patches and rules of
the tools that developed it:

- **The method** is that of Vesanto et al. (2017), at the University of Turku: each letter becomes one of the
  amino-acid symbols that BLAST works with, all documents are compared with each other, and the hits are
  post-processed into passages of reuse. Their implementation is
  [textreuse-blast](https://github.com/avjves/textreuse-blast) (Aleksi Vesanto).
- **The two patches to BLAST** (a uniform scoring matrix, and the statistical parameters for gap costs 3/11) are
  Aleksi Vesanto's changes for textreuse-blast, as documented and ported to BLAST 2.13 by the Helsinki Computational
  History Group (COMHIS) in [text-reuse-blast-custom](https://github.com/COMHIS/text-reuse-blast-custom). textblast
  makes the same changes to BLAST+ 2.17.0.
- **The rules that merge hits into passages** come from three tools:
  - `etl`, the default, re-implements the rule of the ETL behind [Reception Reader](https://receptionreader.com)
    (Rosson et al. 2023; [HPC-HD-TextReuse-ETL](https://github.com/ananth1996/HPC-HD-TextReuse-ETL), Ananth
    Mahadevan);
  - `node_similarity` is adapted from textreuse-blast's clusterizer;
  - `passim` follows the span linking of [passim](https://github.com/dasmiq/passim) (David A. Smith; Smith et al.
    2015).
- **For comparisons,** the `compat-eng` and `compat-fin` profiles use textreuse-blast's letter tables unchanged, and
  the `textreuse-blast` and `comhis` presets use the settings of textreuse-blast and of COMHIS's runs. passim's `locs`
  field and output formats and the ETL's input format are supported as they are.

What textblast contributes is the implementation:

- **Newer:** BLAST+ 2.17.0, built from NCBI's source and checked by textblast itself, or included in the wheels; a
  package that installs with `pip`.
- **Faster:** on one corpus of 39 million letters, on the same 14-core laptop and with textreuse-blast's own
  settings, textblast took 8 minutes from the input texts to passages, where textreuse-blast's pipeline took
  106 minutes: 13 times faster. Most of the difference is that textblast runs many BLAST jobs in parallel, where
  textreuse-blast runs one query at a time. (textreuse-blast ran on BLAST 2.13.0 and textblast on 2.17.0, which on
  its own is only slightly faster.)
- **More to work with:** offsets in the original texts, configurable normalization, masked and walled-off spans,
  page or chapter units, resumable runs on a laptop or a cluster, Parquet tables and exports.

### References

- Vesanto, A., Nivala, A., Rantala, H., Salakoski, T., Salmi, H., & Ginter, F. (2017). Applying BLAST to text reuse
  detection in Finnish newspapers and journals, 1771–1910. In *Proceedings of the NoDaLiDa 2017 Workshop on
  Processing Historical Language* (pp. 54–58). https://aclanthology.org/W17-0510/
- Rosson, D., Mäkelä, E., Vaara, V., Mahadevan, A., Ryan, Y., & Tolonen, M. (2023). Reception Reader: Exploring text
  reuse in early modern British publications. *Journal of Open Humanities Data*, 9, 5.
  https://doi.org/10.5334/johd.101
- Smith, D. A., Cordell, R., & Mullen, A. (2015). Computational methods for uncovering reprinted texts in antebellum
  newspapers. *American Literary History*, 27(3), E1–E15. https://doi.org/10.1093/alh/ajv029

## Citation

If you use textblast, please cite it: [`CITATION.cff`](CITATION.cff) has the details, and GitHub's "Cite this
repository" turns them into BibTeX or APA. Every release is archived on Zenodo: the DOI
[10.5281/zenodo.23729155](https://doi.org/10.5281/zenodo.23729155) stands for all versions, and each version's record has its own DOI.
Please also cite Vesanto et al. (2017), who introduced the method.

## Licence

textblast is under the [MIT licence](LICENSE). The wheels contain programs of NCBI BLAST+, a public-domain work of
the US Government, and libraries under their own licences. textblast also contains textreuse-blast's letter tables,
under textreuse-blast's MIT licence. [`licenses/`](licenses/README.md) has all these licences and says what comes
from each of the tools above.

## Acknowledgements

Funded by the European Union (ERC, SOLEMNE, 101087979, PI Dr. Sven Meeder, Radboud University). Views and opinions expressed are however those of the
author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive
Agency. Neither the European Union nor the granting authority can be held responsible for them.

Developed by Gleb Schmidt (Radboud University, Faculty of Arts, Humanities Lab) with the assistance of Claude
(Anthropic).
