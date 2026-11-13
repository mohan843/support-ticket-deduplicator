# Support Ticket Deduplicator

An NLP mini-project that finds likely duplicate support tickets using TF-IDF vectors and cosine similarity.

## Overview

Duplicate tickets can fragment customer context and increase triage work. This project builds a small searchable ticket corpus, weights terms with TF-IDF, and ranks tickets similar to a new query.

## Problem statement

Given a new support-ticket description, identify existing tickets that may represent the same issue so teams can link, merge, or route them for review.

## Features

- Tokenization and TF-IDF vector construction.
- Cosine similarity ranking.
- Configurable similarity threshold and result count.
- CSV ticket corpus, CLI, and unit tests.
- Standard-library-only implementation.

## Technology stack

- Python 3.10+
- NLP and information retrieval
- TF-IDF and cosine similarity
- CSV, `argparse`, `re`, and `collections`
- `unittest` for automated testing

## Installation

```bash
git clone https://github.com/mohan843/support-ticket-deduplicator.git
cd support-ticket-deduplicator
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

No third-party runtime dependencies are required.

## Execution

```bash
python detector.py "The invoice CSV export is empty" --threshold 0.2
```

## Testing

```bash
python -m unittest discover -s tests -v
```

## Sample output

```text
Matches found: 2
  [2] similarity=0.613 | Invoice export downloads an empty CSV file
  [6] similarity=0.461 | CSV invoice export is blank even though invoices are present
```

## Limitations

The sample corpus is small and lexical. Real support systems need spelling normalization, product metadata, language detection, semantic embeddings, time-aware evaluation, and human review of merges.

## Future improvements

- Add sentence embeddings for paraphrase detection.
- Use product, account, and issue metadata in ranking.
- Calibrate thresholds by duplicate-label validation sets.
- Add streaming ingestion and feedback from support agents.
- Integrate with ticketing systems while preserving customer privacy.
