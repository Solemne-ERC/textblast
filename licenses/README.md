# Third-party licences

textblast itself is under the MIT licence (`../LICENSE`). The platform wheels also contain three programs of NCBI
BLAST+ 2.17.0 (`blastp`, `makeblastdb`, `blastdbcmd`), built from NCBI's source with textblast's two patches. The
source distribution contains no binaries; `textblast engine build` compiles them on your machine from NCBI's source.

The texts in this folder are copied unchanged from the BLAST+ 2.17.0 source distribution.

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
