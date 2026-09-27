# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Team Minutes  
**Team Members:** Sidhant Malik (Team Leader), Pranav Suri, Dhruvaryan Chugh  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We built a two-stage **blocking + gradient-boosted classifier** pipeline. A multi-key inverted index with *forward* (S1→pool) and *reverse* (pool→S1) blocking yields 97.1% candidate recall at full data density. A stage-1 LightGBM model scores ~80 pair features. A stage-2 model then re-scores each pair using how the candidate record competes across **all** Source-1 entities, after which a one-to-one ownership rule and an F0.5-optimised threshold produce the matches. Validation macro F0.5 is 0.978 at full density, and the best public leaderboard score is 0.964.

---

## 2. Methodology

### 2.1 Problem Analysis
EDA on the 2.2M training S1 entities (≈10.3M S2/S3 records) showed:
- **Scale and density:** about 3.46 true matches per S1 (≈1.7 from S2, ≈1.8 from S3); about 5.6% of S1 entities are singletons; about 26% of pool records match no S1 (distractors). No S2/S3 record belongs to more than one S1, so matching is effectively **one-to-one from the record side**.
- **Name noise:** abbreviations and legal-form changes (Pvt/Private, Ltd/Limited, Inc/LLC), typos, word reordering, token-joined names ("jonesamccom"), prefixes/suffixes (e.g. "Services", "The"), and names in **Indic scripts** (Devanagari, Kannada, Odia, …).
- **Address noise:** reordered components, abbreviations (Rd/Road, St/Street), state names vs codes (Haryana/HR, Washington/WA), native-script states, missing PIN/ZIP, care-of (C/O) segments, and **empty addresses** in many S2/S3 records.
- **Hard negatives (decoys):** records at the *same address* with a subtly edited name (an extra syllable, an inserted token, a changed legal form, a bracketed category word) or with the house number shifted or a digit dropped (5025 → 502). There are also **same-name S1 siblings**, which make empty-address records ambiguous.
- **France** appears only in the test set (15% of test S1s), with French address conventions (r./rue, bd, av., "St" = Saint) and legal forms (SARL, SAS, EURL, …).

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage classifier (stacking) + global one-to-one assignment.  
**Core Innovation:** *Competition-aware matching.* Instead of judging each (S1, record) pair in isolation, we compute for each candidate record how well it scores against **every** S1 that retrieved it (probability margin to the best competing S1, rank among S1s, number of high-scoring S1s). Stage 2 learns from these signals, and each record is then assigned to at most one S1. This directly targets decoys and same-name siblings, which dominate the errors of a pairwise model. `rec_p1_margin` is by far the most important stage-2 feature.

---

## 3. Candidate Generation (Blocking)

**Normalisation (before blocking):** Unicode→ASCII transliteration (`unidecode`, word by word for Indic scripts), case/punctuation folding, "&"→"and", collapsing dotted acronyms, a phonetic consonant skeleton (ph→f, c/q/ck→k, v→w, z→s, vowels and h dropped), legal-form extraction (hand-written dictionaries for US, India and France), state-name→code maps (US and India, incl. native-script state names), postcode extraction, C/O and PO-box removal, and French-only address expansion (r→rue, bd→boulevard, st→saint, …, applied only when `country == "France"`). All dictionaries are small, hand-written, and built from the provided records only.

- **Blocking keys used** (hashed to int64, per country and state where available):
  - `n`: single rare name tokens
  - `P`: pairs of name tokens
  - `p`: 5-character prefix of the space-less name
  - `e`: full space-less name (catches joined or domain-style names)
  - `x`: name token × locality word
  - `a`: house number × street word, plus digit-dropped number variants × locality
- **Scoring:** each shared key contributes an IDF-style weight (`log(N / df)`) times a per-key-type weight. Keys more frequent than a cap (150–300 × a multiplier) are skipped, which keeps the index tractable.
- **Forward blocking:** each S1 keeps its top-50 records by score.
- **Reverse blocking:** each S2/S3 record also proposes its top-10 S1 entities. The union rescues records that are buried among many similar candidates on the S1 side (for example, common names). This raised full-density recall from 94.2% (forward only) to 97.1%.
- **Candidate pairs generated (test):** 142,167,097 pairs (82.1 per S1; 86.1M forward, plus 56.0M added by reverse blocking). This is exactly the set our model scores, and it is what `candidate_pairs.tsv` contains.
- **How we ensured true matches were not lost:** recall was always measured at **full density**: sampled validation S1s were queried against the entire 10.3M-record pool, not a subsample, because a subsample hides crowding effects. Validation blocking recall is **97.06%**. Candidate-side statistics (rank of the S1 among all S1s that retrieved the record, gap to the best S1, number of ties) are computed from the blocking scores and passed to the model as features.

---

## 4. Matching Model

**Features used (≈80 in stage 1):**
- **Name features:** RapidFuzz ratio / partial ratio / token-sort / token-set on raw, core (legal form removed), sorted and space-less names; Jaro-Winkler; phonetic-skeleton similarity; token Jaccard; IDF-weighted overlap, and the IDF mass of tokens present in only one side; legal form equal / added / dropped / Jaccard; length features; flags for domain-style names, non-Latin script, and a single rare token.
- **Address features:** token-set and sorted similarity, number-set Jaccard, house-number equal, variant (digit dropped or appended) and absolute difference, postcode equal, state equal, empty address on either side.
- **Decoy detectors:** word extended by a suffix (e.g. "+yn"), first word missing or new, bracketed non-legal category word, long number inserted into the name, partial space-less containment.
- **Blocking features:** blocking score, rank, relative score and gap to the S1's top candidate, number of candidates, reverse score and rank, candidate-side rank among S1s, gap and tie counts, source (S2/S3).

**Model type:** LightGBM (binary, learning rate 0.05, 63 leaves, early stopping; 1,331 trees).
- **Training data:** 80k training S1 entities (5.9M candidate pairs, 4.6% positive), plus competitor pairs: other S1s that also retrieve the same plausible records.
- **Stage 2 (stacking):** a second LightGBM trained on 5-fold out-of-fold stage-1 probabilities. Its extra inputs are:
  - the S1's probability profile: rank, max, gap, sum, count of confident matches, second-best, and rank within source
  - S2↔S3 agreement: whether a candidate matches the other source's confident candidates in name and address
  - **S1-vs-S1 competition:** the best other S1's probability for the same record, the margin, the rank, and the number of high-probability S1s

**Decision rule:** each S2/S3 record is owned by the single S1 with the highest stage-2 probability (one-to-one from the record side). It is output only if that probability is ≥ the threshold.

**Threshold selection method:** grid search maximising macro F0.5 per S1 entity on a held-out validation set of 25k S1 entities scored at full density (chosen threshold 0.65). The model is not trained on validation data. The threshold curve is flat around the optimum (0.60–0.75 are within 0.0005).

---

## 5. Results & Error Analysis

| Version | Main change | Val macro F0.5 (full density) | Public LB |
|---|---|---|---|
| v1 | baseline blocking + LightGBM | 0.938 | 0.927 |
| v3 | richer blocking keys, decoy features | 0.953 | 0.940 |
| v4 | Indic transliteration, candidate-side stats | 0.962 | 0.945 |
| v6 | reverse blocking (top-5) + one-to-one ownership | 0.976 | **0.964** |
| v8 | stage-2 competition model + reverse top-10 | 0.978 | *(v8 score)* |
| v9 | training at test-like density (19% of S1 removed) | 0.977 (on harder, test-like val) | *(v9 score)* |

- **F_0.5 Score (macro):** 0.9779 on validation (stage 2); stage 1 alone 0.9758.
- **Common false positives (wrong merges), 418 of 25k validation entities:**
  - near-identical names where the house number differs by one digit (5025 vs 502) or the street differs
  - S2/S3 records with an **empty address** attached to one of several same-name S1 entities
  - native-script records with only a city-level address
- **Common false negatives (missed matches):**
  - about 2.9% of true pairs are never retrieved as candidates (heavily renamed records such as "Orissa Farm" → "Beloveo" at the same address)
  - 1,915 candidate pairs rejected by the threshold, mostly empty-address records whose name is ambiguous, and records whose name was rewritten or reordered ("Elfreda's Keystone LLC Glass Auto")
- **France (unseen in training):** we measured France's score with one diagnostic submission that emptied all France predictions (the public score fell from 0.964 to 0.830). Solving for France gives F0.5 ≈ 0.95, versus ≈ 0.966 for US and India on the test set. French-specific address and legal-form rules helped but did not fully close the gap.
- **Test-like training (v9):** the test pool has 5.75 S2/S3 records per S1 versus 4.68 in train, which suggests about 19% of test entities have records but no S1 row ("orphan" records that other S1s can wrongly claim). We simulated this by removing 19% of train S1 entities before computing all competition features and training. On this harder validation set, the v8 stage-1 model drops from 0.9758 to 0.9737, the retrained stage-1 model scores 0.9750, and stage 2 scores 0.9768.
- **Validation–leaderboard gap:** about 0.012, and it is present for US and India as well as France. We attribute it to a harder test pool: denser clusters of similar-looking records.

---

## 6. Conclusion
A carefully normalised, multi-key blocking stage with reverse retrieval reaches 97% recall at full scale. Two LightGBM stages then do the matching, and the key signal turned out to be *competition*: whether a record fits this S1 better than every other S1 that also retrieved it. That signal plus one-to-one assignment gave our largest gains against decoys and siblings. The main lessons: evaluate blocking and matching at full data density, and treat entity resolution as an assignment problem rather than independent pair classification. With more compute we would raise the candidate limits, train on more of the 2.2M labelled entities, and add clustering across S2 and S3 records.

---

## Appendix

### A. Code Artefacts
Code is in `code/business_entity_resolution/`: all source in `src/`, plus `README.md` with exact commands and `requirements.txt`. The entry points in order are:

`prepare.py` → `run_normalize.py train|test` → `pipeline.py train` → `train_full.py` → `train_stack.py` → `pipeline.py test`

The last step writes `outputs/submission/matching_results.tsv` and `candidate_pairs.tsv`, then runs the official validator. Everything runs single-process on a 16 GB laptop. A full cycle takes about 9–10 hours; test inference alone takes about 6 hours.

**Libraries:** pandas, numpy, pyarrow, LightGBM, RapidFuzz, unidecode (all MIT/BSD/Apache). **No** pretrained models, external data, gazetteers, libpostal or APIs were used.

### B. Additional Results
- Blocking recall at full density: 92.2% (initial keys) → 94.2% (extended keys) → 96.7% (reverse top-5) → 97.1% (reverse top-10).
- Test output: 5,732,963 predicted matches; 1,626,646 of 1,732,544 S1 entities have at least one match (6.1% predicted singletons). Predicted matches per S1 are similar across countries: US 3.31, India 3.25, France 3.28 (from the v6 run).
- Most important stage-1 features (gain %): `blk_cand_gap` 37.6, `blk_rev_rank` 36.0, `ad_nums_jacc` 5.1, `blk_rev_score` 3.8, `blk_cand_rank` 3.2.
- Most important stage-2 features: `rec_p1_margin` 80.0, `p1` 13.3.