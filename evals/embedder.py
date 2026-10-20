"""A tiny offline embedder for tests and the evaluation suite.

Words with the same meaning share a dimension ("reach", "contact", "email"
-> one concept), so cosine similarity behaves like a real embedding model on
the handful of topics the scenarios use, without a network or model
weights. It stands in for Graphiti's or Mem0's embedder, or any
`EmbeddingRanker(embed)`; it is not a model.
"""

from __future__ import annotations

import hashlib
import re

CONCEPTS = [
    {"reach", "contact", "email", "phone", "call", "message", "notify", "update"},
    {"refund", "return", "money", "back", "reimburse"},
    {"ship", "shipping", "deliver", "delivery", "address", "parcel", "send"},
    {"diet", "food", "eat", "cook", "vegetarian", "vegan", "allergy", "meal"},
    {"size", "fit", "shoe", "wear", "clothes"},
]
DIMS = 64


def embed(texts: list[str]) -> list[list[float]]:
    out = []
    for text in texts:
        vector = [0.0] * DIMS
        for word in re.findall(r"[a-z]+", text.lower()):
            concept = next((i for i, words in enumerate(CONCEPTS) if word in words), None)
            if concept is None:  # other words: their own (hashed) dimension, weighted less
                vector[len(CONCEPTS) + int(hashlib.md5(word.encode()).hexdigest(), 16) % (DIMS - len(CONCEPTS))] += 0.3
            else:
                vector[concept] += 1.0
        out.append(vector)
    return out
