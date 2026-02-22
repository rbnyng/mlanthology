# ML Anthology

An anthology for machine learning research.

## What is this?

ML Anthology indexes papers from ML conferences and journals into a single browsable, searchable site with consistent metadata and human-readable citation keys.

It exists because ML proceedings are fragmented across a dozen different sites with inconsistent interfaces, no unified search, and no persistent identifiers. NLP solved this years ago with [ACL Anthology](https://aclanthology.org). This is the equivalent effort for machine learning. 

Citing an ML paper currently usually means something like getting `10.5555/3295222.3295349` (Vaswani et al.), `NEURIPS2018_5a4be1fa` (this is the NTK paper, but you'd never know), `pmlr-v151-truong22a`, or `7780459` (Resnet, believe it or not) — equally opaque strings, none of which tell you what paper you're looking at. For NeurIPS you have to download a .bib file from each one, open it, and hope you remember which download was which. It sucks.

## Features

- **Human-readable BibTeX keys** — `vaswani2017neurips-attention` instead of opaque hashes or integers
- **Deterministic URLs** — `/venue/year/key`, computable from paper metadata alone
- **Citation export** — BibTeX, plain text, and Markdown with copy buttons
- **Full-text search** — via Pagefind, entirely client-side
- **Author index** — many authors with per-author publication pages
- **Dark and light mode** — follows system preference, looks nice
- **Static site** — fast, no backend, trivially cacheable

## URL scheme

```
mlanthology.org/icml/                                     → latest ICML proceedings
mlanthology.org/icml/2017/                                → ICML 2017 paper listing
mlanthology.org/icml/2017/arora2017icml-generalization    → specific paper at ICML 2017
mlanthology.org/authors/                                  → author index
mlanthology.org/authors/a/                                → authors by last name starting with a
mlanthology.org/authors/a/arora-sanjeev/                  → author page for sanjeev arora
```

Every prefix is a valid page, you can feasibly construct the permalink by guessing if you know the title, year, author, and venue. 

ResNet, for example, is `he2016cvpr-deep` because it appeared at CVPR in 2016, the first author was Kaiming He, and the first word of the title was "Deep".

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

## API Access?

I didn't make one, but every paper page embeds structured metadata as [JSON-LD](https://json-ld.org/) in the document head.

Notice that:

```python
import requests, json, re

def get_paper(key):
    r = requests.get(f"https://mlanthology.org/{key}")
    m = re.search(r'<script type="application/ld\+json">(.*?)</script>', r.text, re.DOTALL)
    return json.loads(m.group(1))

>>> get_paper("neurips/2017/vaswani2017neurips-attention")
{
  "@type": "ScholarlyArticle",
  "headline": "...",
  "author": [...],
  "datePublished": "2017",
  "description": "...",
  ...
}
```

## Data sources

ML Anthology is built on top of open data from:

- [DBLP](https://dblp.org), [Semantic Scholar](https://semanticscholar.org), and [OpenAlex](https://openalex.org) — bibliographic metadata
- [OpenReview](https://openreview.net) — ICLR and TMLR metadata
- [NeurIPS](https://proceedings.neurips.cc) — NeurIPS metadata
- [PMLR](https://proceedings.mlr.press) — ICML, AISTATS, COLT, and other proceedings
- [CVF Open Access](https://openaccess.thecvf.com) — computer vision proceedings

## License

Apache 2.0. See [LICENSE](LICENSE).

Paper metadata is aggregated from the sources above under their respective licenses. ML Anthology does not host PDFs (yet) and papers link to their original sources.
