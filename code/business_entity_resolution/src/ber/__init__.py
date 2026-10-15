"""Business Entity Resolution pipeline (Amazon ML Challenge 2026).

Stages: split -> normalize -> candidates -> features (+ features-hold) -> scorer
-> crossenc -> rerank -> decide -> submit. See README.md for an overview.
"""

__version__ = "0.1.0"
