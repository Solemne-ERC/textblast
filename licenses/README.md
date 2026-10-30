# Third-party licences and sources

textblast itself is under the MIT licence (`../LICENSE`). The platform wheels also contain three programs of NCBI
BLAST+ 2.17.0 (`blastp`, `makeblastdb`, `blastdbcmd`), built from NCBI's source with textblast's two patches. The
source distribution contains no binaries; `textblast engine build` compiles them on your machine from NCBI's source.

## Programs and libraries in the wheels

The texts for these components are copied unchanged from the BLAST+ 2.17.0 source distribution.

| Component | Where | Licence | File |
|---|---|---|---|
| NCBI BLAST+ and the NCBI C++ Toolkit | the three programs | public domain (United States Government Work) | `NCBI-BLAST-public-domain.txt` |
| LMDB (Lightning Memory-Mapped Database) | compiled into the programs | OpenLDAP Public License 2.8 | `LMDB-OLDAP-2.8.txt` |
| lmdb++ (C++ wrapper of LMDB) | compiled into the programs | Unlicense (public domain) | `lmdbxx-Unlicense.txt` |
| PCRE2 | compiled into the programs | BSD-3-Clause, with the PCRE2 exemption | `PCRE2-BSD-3-Clause.txt` |
| Mbed TLS 3.6 | compiled into the programs | Apache-2.0 (Mbed TLS is offered under Apache-2.0 or GPL-2.0-or-later; used here under Apache-2.0) | `Apache-2.0.txt` |
| BitMagic | compiled into the programs | Apache-2.0 | `BitMagic-NOTICE.txt`, `Apache-2.0.txt` |
| bzip2 (libbzip2) | Linux wheel: a copy of the library is included; macOS: the system's | bzip2 licence (BSD-style) | `bzip2.txt` |
| SQLite | Linux wheel: a copy of the library is included; macOS: the system's | public domain | (none required) |

On macOS the programs use only libraries that are part of the operating system. On Linux they also use the
system's zlib, C and C++ libraries, which every manylinux system provides; `auditwheel` copies bzip2 and SQLite
into the wheel (`textblast.libs/`) and lists them in the wheel's `.dist-info/sboms/`.

## The work textblast builds on

textblast is a new implementation of the method of Vesanto et al. (2017) and their tool textreuse-blast; the
README's [Background](../README.md#background) gives the full account and the references. This table lists what
textblast takes from each tool, and under which licence. `textreuse-blast-MIT.txt` is copied unchanged from the
textreuse-blast repository.

| Project | What textblast takes | Licence | File |
|---|---|---|---|
| [textreuse-blast](https://github.com/avjves/textreuse-blast) (Aleksi Vesanto) | the method; the letter tables of the `compat-eng` and `compat-fin` profiles (copied unchanged); the `node_similarity` rule (adapted from its clusterizer); the settings of the `textreuse-blast` preset | MIT | `textreuse-blast-MIT.txt` |
| [text-reuse-blast-custom](https://github.com/COMHIS/text-reuse-blast-custom) (COMHIS), with Aleksi Vesanto's changes to BLAST | the two engine patches: textblast's own code makes the same changes to NCBI's source | none stated (changes to a public-domain work) | — |
| [blast_ecco](https://github.com/villevaara/blast_ecco) (Ville Vaara, COMHIS) | the settings of the `comhis` preset | none stated | — |
| [HPC-HD-TextReuse-ETL](https://github.com/ananth1996/HPC-HD-TextReuse-ETL) (Ananth Mahadevan), the ETL of Reception Reader | the `etl` rule (re-implemented); the `etl` export format | none stated | — |
| [passim](https://github.com/dasmiq/passim) (David A. Smith) | the `passim` rule (re-implemented from its `linkSpans`) and its `mergeSpans`; the `locs` field; the `passim-pairwise` and `passim-clusters` formats | MIT (declared in its `setup.py`) | — |
