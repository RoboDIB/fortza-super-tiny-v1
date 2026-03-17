"""
ZIA — N-gram Context Hash
Parameter-free bigram feature extractor.

Hash constants MUST match firmware/main/ngram.c exactly:
  bucket = (a * 31337 + b * 17) % NGRAM_BUCKETS

No learned parameters. Pure math. Gives the encoder word-level
pattern awareness for free.
"""

import numpy as np

NGRAM_BUCKETS = 32


def zia(indices, n_buckets=NGRAM_BUCKETS):
    """
    Compute bigram hash features from a list of character indices.
    Returns a normalized float32 vector of shape (n_buckets,).

    indices: list of int
    """
    counts = np.zeros(n_buckets, dtype=np.float32)
    for a, b in zip(indices[:-1], indices[1:]):
        bucket = (int(a) * 31337 + int(b) * 17) % n_buckets
        counts[bucket] += 1.0
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def zia_batch(indices_list, n_buckets=NGRAM_BUCKETS):
    """
    Compute ZIA features for a batch.
    indices_list: list of lists of int
    Returns: (B, n_buckets) float32 numpy array
    """
    B = len(indices_list)
    out = np.zeros((B, n_buckets), dtype=np.float32)
    for i, indices in enumerate(indices_list):
        out[i] = zia(indices, n_buckets)
    return out
