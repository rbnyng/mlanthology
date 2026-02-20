# ML Anthology

A unified proceedings platform for machine learning research.

## What is this?

ML Anthology indexes papers from major ML conferences and journals into a single browsable, searchable site with consistent metadata and human-readable citation keys.

It exists because ML proceedings are fragmented across a dozen different sites with inconsistent interfaces, no unified search, and no persistent identifiers. NLP solved this years ago with [ACL Anthology](https://aclanthology.org). This is the equivalent for ML.

## Features

- **Human-readable BibTeX keys** — `vaswani2017neurips-attention` instead of opaque hashes or integers
- **Deterministic URLs** — `/venue/year/key`, computable from paper metadata alone
- **Citation export** — BibTeX, plain text, and Markdown with copy buttons
- **Full-text search** — via Pagefind, entirely client-side
- **Author index** — many authors with per-author publication pages
- **Dark and light mode** — follows system preference
- **Static site** — fast, no backend, trivially cacheable

## URL scheme

```
mlanthology.org/icml/                                     → latest ICML proceedings
mlanthology.org/icml/2017/                                → ICML 2017 paper listing
mlanthology.org/icml/2017/arora2017icml-generalization    → specific paper
mlanthology.org/authors/                                  → author index
mlanthology.org/authors/a/                                → authors starting with a
mlanthology.org/authors/a/arora-sanjeev/                  → author page for arora
```

Every prefix is a valid page, you can feasibly construct the permalink by guessing if you know the title, year, author, and venue.

## BibTeX key format

Keys follow the pattern:

```
{lastname}{year}{venue}-{titlekeyword}
```

- `lastname` — first author's lowercased surname
- `year` — four-digit publication year
- `venue` — lowercased venue abbreviation
- `titlekeyword` — first substantive word from the title (stopwords filtered)

Collisions are resolved with a short suffix. The generation is fully deterministic from the paper metadata.

## Data sources

ML Anthology is built on top of open data from:

- [DBLP](https://dblp.org) — bibliographic metadata (CC0)
- [Semantic Scholar](https://semanticscholar.org) — abstracts and metadata
- [OpenReview](https://openreview.net) — ICLR and TMLR metadata
- [NeurIPS](https://proceedings.neurips.cc) — NeurIPS metadata
- [PMLR](https://proceedings.mlr.press) — ICML, AISTATS, COLT, and other proceedings
- [CVF Open Access](https://openaccess.thecvf.com) — computer vision proceedings

## License

Apache 2.0. See [LICENSE](LICENSE).

Paper metadata is aggregated from the sources above under their respective licenses. ML Anthology does not host PDFs (yet) and all papers link to their original sources.
