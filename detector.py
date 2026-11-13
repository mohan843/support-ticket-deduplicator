"""Detect likely duplicate support tickets with TF-IDF cosine similarity."""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class TfidfDuplicateDetector:
    def fit(self, documents: Sequence[str]) -> "TfidfDuplicateDetector":
        if not documents:
            raise ValueError("At least one document is required")
        self.documents = list(documents)
        self.term_frequencies = [Counter(tokenize(document)) for document in self.documents]
        self.document_frequency = Counter()
        for counts in self.term_frequencies:
            self.document_frequency.update(counts.keys())
        self.size = len(self.documents)
        self.vocabulary = sorted(self.document_frequency)
        return self

    def vectorize(self, text: str) -> dict[str, float]:
        counts = Counter(tokenize(text))
        vector = {}
        for token, count in counts.items():
            if token in self.document_frequency:
                idf = math.log((1 + self.size) / (1 + self.document_frequency[token])) + 1
                vector[token] = count * idf
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {token: value / norm for token, value in vector.items()}

    def similarity(self, left: str, right: str) -> float:
        first, second = self.vectorize(left), self.vectorize(right)
        return sum(first.get(token, 0.0) * value for token, value in second.items())

    def find_duplicates(self, query: str, threshold: float = 0.35, top_k: int = 3) -> list[tuple[int, float, str]]:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        matches = [(index, self.similarity(query, document), document) for index, document in enumerate(self.documents)]
        return sorted((match for match in matches if match[1] >= threshold), key=lambda match: (-match[1], match[0]))[:top_k]


def load_tickets(path: str | Path) -> list[str]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [row["ticket"] for row in csv.DictReader(handle)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find similar support tickets.")
    parser.add_argument("query")
    parser.add_argument("--data", default="data/tickets.csv")
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()
    detector = TfidfDuplicateDetector().fit(load_tickets(args.data))
    matches = detector.find_duplicates(args.query, args.threshold)
    print(f"Matches found: {len(matches)}")
    for index, score, ticket in matches:
        print(f"  [{index}] similarity={score:.3f} | {ticket}")
