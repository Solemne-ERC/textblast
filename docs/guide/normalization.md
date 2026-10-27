# Normalization and BLAST settings

## Why normalize

BLAST compares sequences letter by letter. Two copies of one passage in historical texts rarely agree letter by
letter: one writes `uita`, the other `vita`; one `hec`, the other `hæc`; one `nuntius`, the other `nuncius`; one
spells out what the other abbreviates. Normalization removes the differences that do not matter for reuse, so that
BLAST sees the same letters, while the corpus store keeps the exact place of every letter in the original text. Every
hit is therefore reported in the original text, whatever normalization did.

## Profiles

A profile is a list of steps and a letter table. Choose one with `normalize.profile`.

| profile | letters | |
|---|---:|---|
| `latin-broad` (default) | 20 | the shared rules below, plus: `h` deleted, `ae`/`oe` → `e` inside a word, `c` → `t` before `i` + vowel (`nuncius` → `nuntius`) |
| `latin` | 21 | the shared rules only; `h` is kept |
| `latin-broad-frozen` | 20 | `latin-broad` with an earlier, frequency-based letter table (reproduces runs made with it) |
| `compat-eng`, `compat-fin` | | the encodings of textreuse-blast, character for character, for comparisons with that tool |

The **shared rules** (both Latin profiles):
- lower case; accents removed (`é` → `e`); `æ` → `ae`, `œ` → `oe`, long `ſ` → `s`, ligatures split;
- `j` → `i`, `v` → `u`, `w` → `u`, `y` → `i`, `k` → `c`;
- `&` and `⁊` become the word `et`; a standalone `ł` becomes `uel`;
- a letter with an unresolved abbreviation mark (tilde, macron, overline) is removed, and counted;
- spaces, punctuation and numerals separate words; soft hyphens and `¬` (a hyphen at a line break) disappear;
  `<` and `>` around editorial supplements disappear inside a word (`do<mi>nus` → `dominus`);
- anything else outside a–z is removed and listed in `logs/warnings/unsupported_chars.tsv`.

`latin-broad` is the default because these spelling variants are frequent in Latin texts of every period, and a
borrowing should not be missed because one copy writes `hec` and the other `haec`. The cost is a few false
equivalences (`c`/`t`, a lost `h`), which matter little for passages long enough to be reuse. The rules can be
used for other languages in the Latin script (the quickstart's example is English): they lose distinctions that rarely
matter for matching passages of some length, but they were designed and tested on Latin. Use `latin` when `h`
carries meaning in your texts.

### Your own rules

`normalize.steps` replaces the profile's steps. To add a character rule to `latin-broad`:

```yaml
normalize:
  steps: [spans, {name: graphemes, params: {char_map: {"ð": "d"}}}, fold_jv, fold_wy, fold_kc, delete_h,
          vowel_pairs, ci_vowel]
```

A step of your own is a subclass of `textblast.normalize.Step`, given as
`{class_path: mypackage.MyStep, init_args: {...}}`. Steps must keep the exact map from every letter to its source
span; the built-in steps show how.

### Letter tables

After normalization, each letter is written as one of the 20 amino-acid symbols BLAST knows, so that a text becomes a
protein sequence; `X` stands for barriers. Tables are fixed per profile, not computed per corpus, so the same text is
always encoded the same way. Under the default settings, all letters score alike (see below), so the table barely
matters: two different tables gave 99.98% identical hits. `textblast alphabet suggest --input corpus.jsonl` proposes
a table from the letter frequencies of a corpus, for use with `comp_based_stats 2`.

## BLAST settings

The engine is NCBI BLAST+ 2.17.0 with two patches:
- **a uniform matrix:** every match scores +9 and every mismatch −5, whatever the letters, instead of the protein
  substitution scores of BLOSUM62 (BLAST keeps the matrix's name);
- **gap costs 3/11** with the statistics to match, so that e-values stay meaningful under the uniform matrix.

`textblast engine check` verifies both, and three more properties the pipeline relies on, on tiny synthetic sequences.

### Presets

| preset | `evalue` | `word_size` | `threshold` | `window_size` | `comp_based_stats` | |
|---|---:|---:|---:|---:|---:|---|
| `textblast` (default) | 1e-3 | 6 | 400 | 40 | 0 | |
| `textreuse-blast` | 1e-3 | 6 | 400 | 40 | 2 | the settings of textreuse-blast |
| `comhis` | 1e-9 | 6 | 400 | 40 | 2 | the settings of the COMHIS / ReceptionReader production runs (also `max_target_seqs` 500,000,000) |

Values set under `blast:` override the preset's.

**Why `comp_based_stats 0`.** With `2`, BLAST re-scores each alignment with BLOSUM62 statistics adjusted to the
letter composition of the two sequences. For text, that has three drawbacks: the scores then depend on which
amino-acid symbol each letter happens to be assigned, which has no meaning; the re-alignment can bridge short walls of
`X`; and it takes more than twice as long. On a corpus of early modern Latin texts with references recorded by
editors, `0` found 83% of the Bible quotations against 66% with `2`, with hits that end nearer the reused text. Most of
the hits found only by `2` were judged not to be reuse. Most of the extra hits found by `0` are short; many are
formulae (see [Recipes](recipes.md#formulae-and-stock-phrases)).

**Why `evalue 1e-3`.** It is textreuse-blast's value. `1e-5` lost quotations an expert wanted found;
`1e-2` added half as many hits again, nearly all tiny formulae, for very few more true quotations.

**Why `window_size 40`.** Two seeds within 40 letters start an extension. `0` (one seed is enough) took 3.4 times as
long and found nothing more of value.
