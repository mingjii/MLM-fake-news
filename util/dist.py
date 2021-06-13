import numpy as np


def length(p: float, max_span_len: int):
    """Geometric distribution follow SpanBERT's instruction."""
    while True:
        sample = int(np.random.geometric(p=p, size=1))
        if sample <= max_span_len:
            return sample


def mask(p: float):
    """Decide whether to mask a token."""
    return int(np.random.binomial(n=1, p=p))
