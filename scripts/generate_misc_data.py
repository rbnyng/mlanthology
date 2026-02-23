#!/usr/bin/env python3
"""Generate loose JSON files for the misc (Notable Works) section.

One-time script to seed data/misc/ with foundational documents.
Each paper gets its own .json file named after its bibtex key.
"""

import json
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from adapters.common import slugify_author, make_bibtex_key, resolve_bibtex_collisions

OUTPUT_DIR = Path(_project_root) / "data" / "misc"

papers = [
    # === The Foundations ===
    {
        "title": "An Essay towards Solving a Problem in the Doctrine of Chances",
        "authors": [{"given": "Thomas", "family": "Bayes"}],
        "year": "1763",
        "venue_name": "Philosophical Transactions of the Royal Society",
        "pages": "370-418",
        "doi": "10.1098/rstl.1763.0053",
    },
    {
        "title": "Theoria Motus Corporum Coelestium",
        "authors": [{"given": "Carl Friedrich", "family": "Gauss"}],
        "year": "1809",
        "venue_name": "Perthes et Besser, Hamburg",
    },
    {
        "title": "A Logical Calculus of the Ideas Immanent in Nervous Activity",
        "authors": [
            {"given": "Warren S.", "family": "McCulloch"},
            {"given": "Walter", "family": "Pitts"},
        ],
        "year": "1943",
        "venue_name": "Bulletin of Mathematical Biophysics",
        "volume": "5",
        "pages": "115-133",
        "doi": "10.1007/BF02478259",
    },
    {
        "title": "A Mathematical Theory of Communication",
        "authors": [{"given": "Claude E.", "family": "Shannon"}],
        "year": "1948",
        "venue_name": "Bell System Technical Journal",
        "volume": "27",
        "pages": "379-423, 623-656",
        "pdf_url": "https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf",
        "doi": "10.1002/j.1538-7305.1948.tb01338.x",
    },
    {
        "title": "Computing Machinery and Intelligence",
        "authors": [{"given": "Alan M.", "family": "Turing"}],
        "year": "1950",
        "venue_name": "Mind",
        "volume": "59",
        "number": "236",
        "pages": "433-460",
        "doi": "10.1093/mind/LIX.236.433",
    },
    # === The Neural Thread ===
    {
        "title": "The Organization of Behavior: A Neuropsychological Theory",
        "authors": [{"given": "Donald O.", "family": "Hebb"}],
        "year": "1949",
        "venue_name": "Wiley, New York",
    },
    {
        "title": "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain",
        "authors": [{"given": "Frank", "family": "Rosenblatt"}],
        "year": "1958",
        "venue_name": "Psychological Review",
        "volume": "65",
        "number": "6",
        "pages": "386-408",
        "doi": "10.1037/h0042519",
    },
    {
        "title": "Receptive Fields, Binocular Interaction and Functional Architecture in the Cat's Visual Cortex",
        "authors": [
            {"given": "David H.", "family": "Hubel"},
            {"given": "Torsten N.", "family": "Wiesel"},
        ],
        "year": "1962",
        "venue_name": "Journal of Physiology",
        "volume": "160",
        "number": "1",
        "pages": "106-154",
        "doi": "10.1113/jphysiol.1962.sp006837",
    },
    {
        "title": "Perceptrons: An Introduction to Computational Geometry",
        "authors": [
            {"given": "Marvin", "family": "Minsky"},
            {"given": "Seymour", "family": "Papert"},
        ],
        "year": "1969",
        "venue_name": "MIT Press",
    },
    {
        "title": "Neocognitron: A Self-organizing Neural Network Model for a Mechanism of Pattern Recognition Unaffected by Shift in Position",
        "authors": [{"given": "Kunihiko", "family": "Fukushima"}],
        "year": "1980",
        "venue_name": "Biological Cybernetics",
        "volume": "36",
        "number": "4",
        "pages": "193-202",
        "doi": "10.1007/BF00344251",
    },
    {
        "title": "Neural Networks and Physical Systems with Emergent Collective Computational Abilities",
        "authors": [{"given": "John J.", "family": "Hopfield"}],
        "year": "1982",
        "venue_name": "Proceedings of the National Academy of Sciences",
        "volume": "79",
        "number": "8",
        "pages": "2554-2558",
        "doi": "10.1073/pnas.79.8.2554",
    },
    {
        "title": "Learning Representations by Back-propagating Errors",
        "authors": [
            {"given": "David E.", "family": "Rumelhart"},
            {"given": "Geoffrey E.", "family": "Hinton"},
            {"given": "Ronald J.", "family": "Williams"},
        ],
        "year": "1986",
        "venue_name": "Nature",
        "volume": "323",
        "pages": "533-536",
        "doi": "10.1038/323533a0",
    },
    {
        "title": "Long Short-Term Memory",
        "authors": [
            {"given": "Sepp", "family": "Hochreiter"},
            {"given": "Jürgen", "family": "Schmidhuber"},
        ],
        "year": "1997",
        "venue_name": "Neural Computation",
        "volume": "9",
        "number": "8",
        "pages": "1735-1780",
        "doi": "10.1162/neco.1997.9.8.1735",
    },
    {
        "title": "Gradient-Based Learning Applied to Document Recognition",
        "authors": [
            {"given": "Yann", "family": "LeCun"},
            {"given": "Léon", "family": "Bottou"},
            {"given": "Yoshua", "family": "Bengio"},
            {"given": "Patrick", "family": "Haffner"},
        ],
        "year": "1998",
        "venue_name": "Proceedings of the IEEE",
        "volume": "86",
        "number": "11",
        "pages": "2278-2324",
        "doi": "10.1109/5.726791",
        "pdf_url": "http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf",
    },
    # === The Statistical / Learning Theory Thread ===
    {
        "title": "A Stochastic Approximation Method",
        "authors": [
            {"given": "Herbert", "family": "Robbins"},
            {"given": "Sutton", "family": "Monro"},
        ],
        "year": "1951",
        "venue_name": "Annals of Mathematical Statistics",
        "volume": "22",
        "number": "3",
        "pages": "400-407",
        "doi": "10.1214/aoms/1177729586",
    },
    {
        "title": "A Formal Theory of Inductive Inference, Part I",
        "authors": [{"given": "Ray J.", "family": "Solomonoff"}],
        "year": "1964",
        "venue_name": "Information and Control",
        "volume": "7",
        "number": "1",
        "pages": "1-22",
        "doi": "10.1016/S0019-9958(64)90223-2",
    },
    {
        "title": "On the Uniform Convergence of Relative Frequencies of Events to Their Probabilities",
        "authors": [
            {"given": "Vladimir N.", "family": "Vapnik"},
            {"given": "Alexey Ya.", "family": "Chervonenkis"},
        ],
        "year": "1971",
        "venue_name": "Theory of Probability and Its Applications",
        "volume": "16",
        "number": "2",
        "pages": "264-280",
        "doi": "10.1137/1116025",
    },
    {
        "title": "Maximum Likelihood from Incomplete Data via the EM Algorithm",
        "authors": [
            {"given": "Arthur P.", "family": "Dempster"},
            {"given": "Nan M.", "family": "Laird"},
            {"given": "Donald B.", "family": "Rubin"},
        ],
        "year": "1977",
        "venue_name": "Journal of the Royal Statistical Society, Series B",
        "volume": "39",
        "number": "1",
        "pages": "1-38",
        "doi": "10.1111/j.2517-6161.1977.tb01600.x",
    },
    {
        "title": "Modeling by Shortest Data Description",
        "authors": [{"given": "Jorma", "family": "Rissanen"}],
        "year": "1978",
        "venue_name": "Automatica",
        "volume": "14",
        "number": "5",
        "pages": "465-471",
        "doi": "10.1016/0005-1098(78)90005-5",
    },
    {
        "title": "A Theory of the Learnable",
        "authors": [{"given": "Leslie G.", "family": "Valiant"}],
        "year": "1984",
        "venue_name": "Communications of the ACM",
        "volume": "27",
        "number": "11",
        "pages": "1134-1142",
        "doi": "10.1145/1968.1972",
    },
    {
        "title": "The Lack of A Priori Distinctions Between Learning Algorithms",
        "authors": [{"given": "David H.", "family": "Wolpert"}],
        "year": "1996",
        "venue_name": "Neural Computation",
        "volume": "8",
        "number": "7",
        "pages": "1341-1390",
        "doi": "10.1162/neco.1996.8.7.1341",
    },
    {
        "title": "Natural Gradient Works Efficiently in Learning",
        "authors": [{"given": "Shun-ichi", "family": "Amari"}],
        "year": "1998",
        "venue_name": "Neural Computation",
        "volume": "10",
        "number": "2",
        "pages": "251-276",
        "doi": "10.1162/089976698300017746",
    },
    # === The Reinforcement Learning Lineage ===
    {
        "title": "Dynamic Programming",
        "authors": [{"given": "Richard", "family": "Bellman"}],
        "year": "1957",
        "venue_name": "Princeton University Press",
    },
    {
        "title": "Some Studies in Machine Learning Using the Game of Checkers",
        "authors": [{"given": "Arthur L.", "family": "Samuel"}],
        "year": "1959",
        "venue_name": "IBM Journal of Research and Development",
        "volume": "3",
        "number": "3",
        "pages": "210-229",
        "doi": "10.1147/rd.33.0210",
    },
    {
        "title": "Learning to Predict by the Methods of Temporal Differences",
        "authors": [{"given": "Richard S.", "family": "Sutton"}],
        "year": "1988",
        "venue_name": "Machine Learning",
        "volume": "3",
        "number": "1",
        "pages": "9-44",
        "doi": "10.1007/BF00115009",
    },
    {
        "title": "Learning from Delayed Rewards",
        "authors": [{"given": "Christopher J. C. H.", "family": "Watkins"}],
        "year": "1989",
        "venue_name": "PhD thesis, University of Cambridge",
    },
    # === The Information / Coding Thread ===
    {
        "title": "Possible Principles Underlying the Transformations of Sensory Messages",
        "authors": [{"given": "Horace B.", "family": "Barlow"}],
        "year": "1961",
        "venue_name": "Sensory Communication, MIT Press",
        "pages": "217-234",
    },
    {
        "title": "Three Approaches to the Quantitative Definition of Information",
        "authors": [{"given": "Andrey N.", "family": "Kolmogorov"}],
        "year": "1965",
        "venue_name": "Problems of Information Transmission",
        "volume": "1",
        "number": "1",
        "pages": "1-7",
    },
    {
        "title": "Bayesian Learning for Neural Networks",
        "authors": [{"given": "Radford M.", "family": "Neal"}],
        "year": "1996",
        "venue_name": "PhD thesis, University of Toronto",
    },
    # === The AI Critiques and Polemics ===
    {
        "title": "Alchemy and Artificial Intelligence",
        "authors": [{"given": "Hubert L.", "family": "Dreyfus"}],
        "year": "1965",
        "venue_name": "RAND Corporation",
    },
    {
        "title": "Artificial Intelligence: A General Survey",
        "authors": [{"given": "James", "family": "Lighthill"}],
        "year": "1973",
        "venue_name": "UK Science Research Council",
    },
    {
        "title": "Intelligence without Representation",
        "authors": [{"given": "Rodney A.", "family": "Brooks"}],
        "year": "1991",
        "venue_name": "Artificial Intelligence",
        "volume": "47",
        "pages": "139-159",
        "doi": "10.1016/0004-3702(91)90053-M",
    },
    # === The Proposals and Founding Documents ===
    {
        "title": "A Proposal for the Dartmouth Summer Research Project on Artificial Intelligence",
        "authors": [
            {"given": "John", "family": "McCarthy"},
            {"given": "Marvin L.", "family": "Minsky"},
            {"given": "Nathaniel", "family": "Rochester"},
            {"given": "Claude E.", "family": "Shannon"},
        ],
        "year": "1955",
        "venue_name": "Dartmouth College",
    },
    {
        "title": "Pandemonium: A Paradigm for Learning",
        "authors": [{"given": "Oliver G.", "family": "Selfridge"}],
        "year": "1959",
        "venue_name": "Mechanisation of Thought Processes, HMSO London",
        "pages": "511-529",
    },
    {
        "title": "Adaptation in Natural and Artificial Systems",
        "authors": [{"given": "John H.", "family": "Holland"}],
        "year": "1975",
        "venue_name": "University of Michigan Press",
    },
    {
        "title": "Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference",
        "authors": [{"given": "Judea", "family": "Pearl"}],
        "year": "1988",
        "venue_name": "Morgan Kaufmann",
    },
    # === The Blog Posts and Informal Publications ===
    {
        "title": "Curious Model-Building Control Systems",
        "authors": [{"given": "Jürgen", "family": "Schmidhuber"}],
        "year": "1991",
        "venue_name": "Technical Report FKI-150-91, TU Munich",
    },
    {
        "title": "The Unreasonable Effectiveness of Recurrent Neural Networks",
        "authors": [{"given": "Andrej", "family": "Karpathy"}],
        "year": "2015",
        "venue_name": "Blog post",
        "venue_url": "https://karpathy.github.io/2015/05/21/rnn-effectiveness/",
    },
    {
        "title": "Understanding LSTM Networks",
        "authors": [{"given": "Christopher", "family": "Olah"}],
        "year": "2015",
        "venue_name": "Blog post",
        "venue_url": "https://colah.github.io/posts/2015-08-Understanding-LSTMs/",
    },
    {
        "title": "The Bitter Lesson",
        "authors": [{"given": "Richard S.", "family": "Sutton"}],
        "year": "2019",
        "venue_name": "Blog post",
        "venue_url": "http://www.incompleteideas.net/IncIdeas/BitterLesson.html",
    },
]


def generate():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate bibtex keys
    keys = []
    for p in papers:
        first_family = p["authors"][0]["family"]
        keys.append(make_bibtex_key(first_family, p["year"], "misc", p["title"]))

    keys = resolve_bibtex_collisions(keys)

    for key, p in zip(keys, papers):
        authors = []
        for a in p["authors"]:
            slug = slugify_author(a)
            authors.append({**a, "slug": slug})

        paper = {
            "bibtex_key": key,
            "title": p["title"],
            "authors": authors,
            "year": p["year"],
            "venue": "misc",
            "venue_name": p.get("venue_name", ""),
            "venue_type": "misc",
            "volume": p.get("volume", ""),
            "number": p.get("number", ""),
            "pages": p.get("pages", ""),
            "abstract": p.get("abstract", ""),
            "pdf_url": p.get("pdf_url", ""),
            "venue_url": p.get("venue_url", ""),
            "doi": p.get("doi", ""),
            "openreview_url": "",
            "code_url": "",
            "source": "manual",
            "source_id": "",
        }

        out_path = OUTPUT_DIR / f"{key}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(paper, f, ensure_ascii=False, indent=2)
        print(f"  {key}.json")

    print(f"\nGenerated {len(keys)} files in {OUTPUT_DIR}")


if __name__ == "__main__":
    generate()
