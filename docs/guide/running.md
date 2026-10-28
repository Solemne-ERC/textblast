# Running

## The engine

```
textblast engine info           # which engine is used: version, fingerprint, where it was found, last check
textblast engine check          # test it on tiny synthetic sequences (seconds); a run does this once by itself
textblast engine build          # download NCBI's source, check it, patch it, compile it (5–45 minutes)
```

A wheel installed with `pip` carries the engine, so `engine build` is only needed after installing from source (on
platforms without a wheel). The engine is looked for in this order:
1. `--blast-bin DIR` (or `engine.blast_bin` in the configuration): a directory with `blastp`, `makeblastdb` and
   `blastdbcmd`;
2. the environment variable `TEXTBLAST_BIN`, the same;
3. the engine bundled in the installed package;
4. the engines built by `engine build`, in `~/.cache/textblast` (or `TEXTBLAST_CACHE`).

The bundled engine comes before the cache because it is the one the installed version was built and tested with. To
use one you built yourself instead, pass its directory with `--blast-bin` (`engine build` prints it). Every run
records which engine it used (version, fingerprint, place and how it was found) in `blast/plan.json` and the
manifest.

`engine build` options: `--version 2.13.0` builds the reference engine; `--openmp on|off` (default: on for Linux, off
for macOS, whose compiler has none); `--jobs N` limits parallel compilation.

## Stages

```
textblast run --config run.yaml       # all of it: prepare → plan → blast → merge → post → export
```

or one stage at a time:

| command | does | writes |
|---|---|---|
| `prepare --config run.yaml` | reads, checks, normalizes and encodes the input | `corpus/` |
| `plan --config run.yaml` | splits the documents into blocks, makes the BLAST databases' list and the jobs | `blast/plan.json`, `blast/jobs.tsv` |
| `blast --run-dir DIR` | runs every job not done yet, the longest first | `blast/jobs/` |
| `merge --run-dir DIR` | gathers the jobs' hits | `tables/hsps.parquet` |
| `post --config run.yaml` | joins, defragments, annotates, filters, clusters | the other tables in `tables/` |
| `export --config run.yaml --format pairs_flat` | converts | `exports/` |
| `inspect --run-dir DIR` | shows the state of every stage, also while a run is going | |

`post` and `export` also accept `--run-dir DIR` alone, with the configuration the run was made with.
`-q` shows only warnings and errors, `-v` adds detail and tracebacks.

Each stage is skipped when its inputs and settings have not changed, so `run` can always be started again. A stage
whose input changed is redone, and so is everything after it.

## Resuming

Each BLAST job writes its result in one step, with a hash of everything it depends on (documents, letter table,
settings, engine). After an interruption, `blast` (or `run`) does only the jobs that are missing or out of date. If
some jobs fail, the others are kept: the error is in `logs/jobs/`, and running `blast` again retries the failed ones.
`merge --partial` merges what is done, for a first look.

## Computing clusters (Slurm and others)

The jobs are independent, so they can run as an array job: one array task per job.

```bash
textblast prepare --config run.yaml
textblast plan --config run.yaml
textblast engine check                       # once: run-job refuses an unchecked engine
N=$(( $(wc -l < runs/mycorpus/blast/jobs.tsv) - 1 ))
sbatch --array=1-$N --cpus-per-task=1 --mem=2G --wrap \
    'textblast run-job --run-dir runs/mycorpus --job $SLURM_ARRAY_TASK_ID'
# when the array has finished:
textblast inspect --run-dir runs/mycorpus    # all jobs done?
textblast merge --run-dir runs/mycorpus
textblast post --config run.yaml
```

Job ids run from 1 to the number of lines in `blast/jobs.tsv` (minus the header). `run-job` does one job and nothing
else, even if it was done before (the result is the same); `blast --jobs 1-20,35` runs a selection on one machine. The run directory must be on a disk every node sees. Memory: about
0.5 GB per BLAST job on a corpus of 39 million letters.

## Workers and threads

`blast.workers` jobs run at once, each with `blast.threads_per_worker` threads. Use **one thread per worker** and as
many workers as there are cores to spare. Measured on the full 39-million-letter corpus:

| machine | workers × threads | BLAST time |
|---|---|---:|
| laptop, Apple M4 Pro (10 performance cores) | 10 × 1 | 183 s |
| server, 2 × AMD EPYC 7302 (32 cores, 64 threads) | 8 × 1 | 684 s |
| | 16 × 1 | 351 s |
| | 32 × 1 | 206 s |
| | 64 × 1 | 150 s |
| | 16 × 2 | 307 s |
| | 8 × 4 | 457 s |

- Time falls in proportion to workers up to the number of physical cores. Hyperthreads (two per core) add about a
  quarter more: use them when the machine is yours alone.
- On a shared machine, use the number of cores that are actually free: workers that wait for a core only add
  overhead.
- Threads per worker help only with `comp_based_stats 2` (16 × 2 was 4% faster than 32 × 1), and only in an engine
  built with OpenMP (`engine build` on Linux). The engine in the Linux wheel is built without it.
- A fast core matters: one core of the laptop did the work of about 2.8 cores of the server.

`post` and the exports run mostly in one process; `workers` (top level) sets their helper processes. The `pairs_flat`
export with excerpts needs about 0.8 GB per worker on the corpus above: lower `workers` if memory is short.
Keeping the run directory on a network disk made no measurable difference.

## Logs

- the console: a few lines per stage, with progress measured in work (letters, letter pairs), not in files;
- `logs/run.log`: the same, with more detail; `logs/events.jsonl`: one JSON record per event, for scripts;
- `logs/warnings/*.tsv`: summaries of what normalization removed (`unsupported_chars.tsv`,
  `exclusions_by_reason.tsv`), and other warnings that concern many documents;
- `logs/jobs/`: the errors of failed BLAST jobs.
