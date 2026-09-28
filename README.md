# Business Entity Resolution — Team Minutes (Amazon ML Challenge 2026)

Reproduces `matching_results.tsv` and `candidate_pairs.tsv` from the provided train/test TSVs.
Uses only the provided data and open-source, pure-algorithm libraries (no external data, APIs,
gazetteers or pretrained models).

## Environment
- Python 3.12, 16 GB RAM is enough (tested on a Windows 11 laptop, single process).
- `pip install -r requirements.txt`

## Data layout
Place the challenge data as:
```
business_entity_resolution/
  data/dataset/train/train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
  data/dataset/test/test_source1.tsv,  test_source2.tsv,  test_source3.tsv
  src/ ...
```
Intermediate files are written to `data/parquet/`; models and outputs to `outputs/`.

## Run end-to-end (from the `business_entity_resolution/` folder)
```
python src/prepare.py                 # TSV -> parquet, ground-truth mapping
python src/run_normalize.py train     # normalise names/addresses (streaming, ~1h)
python src/run_normalize.py test
python src/pipeline.py train          # blocking at full density + pair features (~2-3h)
python src/train_full.py              # stage-1 LightGBM (~10 min)
python src/train_stack.py             # stage-1 OOF + stage-2 LightGBM (~35 min)
python src/pipeline.py test           # blocking + scoring + 1:1 assignment on test (~6h)
```
Outputs: `outputs/submission/matching_results.tsv` and `outputs/submission/candidate_pairs.tsv`.
The final step also runs `utils/validate_submission.py` on both files.

Training simulates the test's density by removing 19% of train S1 entities (env `DROP`, default 0.19; `DROP=0` disables it).

Optional: `python src/pipeline.py train reusefwd` / `python src/pipeline.py test reuse` reuse the
cached candidate sets in `data/parquet/` instead of recomputing blocking.

## Source files (`src/`)
| File | Purpose |
|---|---|
| `prepare.py` | Loads TSVs, writes parquet tables and ground-truth pairs |
| `normalize.py`, `run_normalize.py` | Name/address normalisation (transliteration, legal forms, state codes, French rules) |
| `blocking.py` | Multi-key inverted index (`KeyIndex`) with frequency caps and IDF-weighted scores |
| `features.py` | ~80 pair features (fuzzy name/address, token IDF, decoy detectors, blocking stats) |
| `pipeline.py` | Forward + reverse blocking, candidate statistics, train/val feature building, test entry point |
| `train_full.py` | Stage-1 LightGBM + threshold search on validation |
| `stack.py`, `train_stack.py` | Stage-2 stacking features (competition between S1s, S2-S3 agreement) and training |
| `predict.py` | Test inference: features → stage 1 → stage 2 → one-to-one ownership → TSVs |
| `metrics.py` | Macro F0.5 per Source-1 entity |
| `explog.py` | Experiment log (`outputs/experiments.csv`) |
| `diagnose.py`, `deep_diag.py`, `decide_eval.py` | Error analysis utilities (not needed to reproduce) |
