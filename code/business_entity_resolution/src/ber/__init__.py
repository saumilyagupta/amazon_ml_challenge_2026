"""ber: Business Entity Resolution pipeline (ML Challenge 2026).

Stages (entry points in src/):  prepare.py -> block.py -> features.py -> train.py -> predict.py -> validate.py
Matcher variants are YAML configs in src/configs/ (v1, v2a, H = hybrid of v1 and v2a, v2b = the previous submission:
union v2 + explainer features + density-robust 2-fold 2-stage LightGBM + R10c set-decoder; v3 = THE SUBMITTED MATCHER: v2b + offset-
conditioned decoy features (v3decoy) + second-explainer features (branch/) + France-gated France pack (francepack / packrecords /
packfeats) + own-probability exclusivity and one-owner in the decoder (setdecoder); trained models in resources/models/v3).
All heavy code is CPU (polars / numpy / rapidfuzz / LightGBM); the dense retriever (multilingual-e5-small) can run on GPU or CPU.
Licence notes: anyascii (ISC) is the only transliteration/folding library used (the branch explainer folds with the standard
library's unicodedata); no GPL transliteration library anywhere.
No external services, lookups or telemetry are used at any point.
"""
__version__ = '3.0.0'
