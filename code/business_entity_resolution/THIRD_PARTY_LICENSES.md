# Third-party models and libraries used by this package

The team's own code (`src/`, including the second explainer `src/ber/branch/` written by a team member) is released under the MIT licence in
`LICENSE`.  Every model that produced the submitted file is MIT or Apache-2.0 licensed and far below the 8-billion-parameter limit (the largest has 568M
parameters).  No external data, lookup or model API was used.  The package's only network access is the one-time download of the public
base encoder into the local HF cache; separately, five context cross-encoders of the later layers were trained and scored on rented cloud
GPUs (Modal), which is compute only.

| component | licence | parameters | where it is used |
|---|---|---|---|
| `intfloat/multilingual-e5-small` (Hugging Face, snapshot 614241f6) | MIT | 118M | zero-shot dense blocking channel; base of the two fine-tuned encoders |
| `src/resources/models/e5s_ft_trainsplit`, `e5s_ft_all` (multilingual-e5-small fine-tuned on the provided training labels) | MIT (derived work, released under the same licence) | 118M each | fine-tuned dense member of blocking union v2 (train split / test split) |
| `FacebookAI/xlm-roberta-base` (Hugging Face) fine-tuned as a pairwise cross-encoder | MIT | 278M | ensemble member E06 of the submitted file (uncertain band only; scored on GPU; the fine-tuned weights, 3 x 1.1 GB, are NOT shipped -- the stored test logits are, `src/resources/ens8_frozen/audit/ce_xw_*.parquet`) |
| `FacebookAI/xlm-roberta-base` further fine-tunes (CE-XL and its variants, FR-SYNTH) | MIT | 278M each | CE-NC correction, US / India head columns and vetoes, France veto L10 (weights not shipped) |
| `jhu-clsp/mmBERT-base` (Hugging Face) fine-tunes | MIT | 308M each | cross-encoder M1 (US / India head column), the sibling-aware context cross-encoders (heads and vetoes), French cross-encoders FRCE1 / FRCE2 (France vetoes) (weights not shipped) |
| `BAAI/bge-reranker-v2-m3` (Hugging Face) fine-tune | Apache-2.0 | 568M | cross-encoder A2, a score column of the US / India head (weights not shipped) |
| AI4Bharat IndicXlit v1.0 (Indic -> Latin transliteration) | MIT (`src/resources/LICENSE.IndicXlit.txt`) | ~11M | used once offline to propose 165 of the 1,534 entries of `src/resources/translit_dictionary.tsv`; only the dictionary is shipped |
| LightGBM 4.7.0 | MIT | trees (< 15 MB per model) | matchers v1 / v2b / v3, set decoders, E06 stage 3, E13 graph specialist, number-residual seeds, empty-address specialist |
| polars, numpy, scipy, scikit-learn, pandas, pyarrow, PyYAML, tqdm | MIT / BSD-3 / Apache-2.0 | - | infrastructure |
| torch, transformers, sentence-transformers, tokenizers, safetensors, huggingface-hub | BSD-3 / Apache-2.0 | - | encoders and the cross-encoder |
| rapidfuzz | MIT | - | string similarities |
| anyascii | ISC | - | accent / script folding (no GPL `unidecode` anywhere in the shipped code) |
| sparse_dot_topn | Apache-2.0 | - | sparse top-k for the TF-IDF channels |

## MIT licence text (applies to multilingual-e5-small, xlm-roberta-base, mmBERT-base, IndicXlit, LightGBM, rapidfuzz, polars, and this package's code)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions: the above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The copyright holders of the third-party components are their respective authors (Microsoft / the E5 authors; Meta AI / the XLM-R authors;
AI4Bharat; the LightGBM, rapidfuzz and polars maintainers); their notices are distributed with the original packages and model cards.
