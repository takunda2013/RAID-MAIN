# XGBoost Probability Confidence Issue

## Problem Summary

The binary XGBoost classifier is correctly separating `Human` essays from `AI`
essays, where `AI` includes both original AI text and paraphrased AI text.
However, the predicted probabilities are usually extreme:

- Human examples often receive `prob_human` near `0.99`.
- AI examples often receive `prob_ai` near `0.99`.
- This happens even when the train, validation, and test splits are not exact
  duplicates.

This is a problem because the probability is being treated as a meaningful
confidence score. For the LIME auditor, extreme probabilities make explanations
less useful: small text perturbations may not move the prediction enough, so
LIME mostly sees a saturated classifier instead of a model with informative
local uncertainty.

## Current Evidence

The latest saved training output shows very strong classification performance:

- Test accuracy: `99.24%`
- Raw AUC: `0.9957`
- Calibrated AUC: `0.9957`
- False positive rate: `0.84%`
- False negative rate: `0.63%`

The probability distribution is highly saturated:

- Raw probabilities have only `217` unique values across `44,867` test rows.
- About `60.4%` of rows have `prob_ai <= 0.01`.
- About `36.0%` of rows have `prob_ai >= 0.99`.
- Only about `1.6%` of rows fall in the uncertain range `0.1 < prob_ai < 0.9`.

The most repeated raw probabilities are:

- `0.007076`, repeated about `27,017` times.
- `0.992959`, repeated about `16,061` times.

This does not appear to be mainly caused by duplicate feature rows:

- `unseen`: `44,867` rows, `44,864` unique feature rows.
- `test`: `6,087` rows, `6,086` unique feature rows.
- `val`: `6,083` rows, `6,083` unique feature rows.

The feature importance files show a stronger clue:

- Only `11` features are used by the trained XGBoost model.
- All used features are `semantic_*` RoBERTa embedding dimensions.
- No statistical surprisal features appear in the used feature importance list.

This means the current model is not behaving like a balanced semantic plus
statistical detector. It is mostly acting as a shallow tree classifier over a
small number of RoBERTa embedding dimensions.

## Important Configuration Issue

The header comment in `train_xgboost_binary-copy.py` says:

- Train on `unseen`
- Validate/calibrate on `val`
- Final test on `test`

But the actual config currently says:

```python
"train_split": "test",
"val_split":   "val",
"test_split":  "unseen",
```

This mismatch does not automatically explain the extreme probabilities, but it
does make the experiment harder to reason about. The first cleanup step should
be to make the comments, config, file names, and metrics agree.

## Likely Root Cause

The most likely root cause is not a simple train/test duplication bug. The
stronger explanation is:

1. The RoBERTa contrastive embeddings are already highly separable by label.
2. XGBoost only needs a few embedding dimensions to split the data almost
   perfectly.
3. Tree models produce piecewise-constant probabilities, so many different
   essays land in the same leaves and receive identical probability values.
4. Because the classes are very separable, those leaves have class ratios close
   to all-human or all-AI, producing `0.99`-style probabilities.
5. Platt calibration does not fix this here. In the saved results, calibrated
   probabilities are even more extreme than raw probabilities.

In short: the model is probably accurate, but its probability output is not a
well-behaved uncertainty estimate for auditing.

## Secondary Risks To Check

### 1. Semantic Embedding Leakage

The RoBERTa contrastive model may have learned very strong dataset/source
signals rather than general writing signals. If essays from different generators,
prompts, sources, or preprocessing paths correlate strongly with labels, the
embedding can make the downstream task too easy.

Recommended check:

- Evaluate on a completely external dataset that was not used in RoBERTa
  training, XGBoost training, calibration, threshold tuning, or feature
  engineering decisions.
- Prefer a source-level split, not just a random essay-level split.
- Example: hold out entire prompts, assignment types, schools, AI generators,
  paraphrasers, or collection batches.

### 2. Split Naming And Experiment Confusion

The model currently trains on a split named `test` and evaluates on `unseen`.
That may be intentional, but it should be renamed or documented clearly.

Recommended check:

- Decide the canonical split policy.
- Use names like `train`, `calibration`, and `final_test`.
- Save these exact names into `training_config.json`.

### 3. Calibration Set Too Similar Or Too Easy

Calibration can only improve probabilities if the calibration data represents
the uncertainty expected at inference time. If the calibration set is also very
separable, calibration will confidently map scores closer to `0` or `1`.

Recommended check:

- Report Brier score, log loss, expected calibration error, and reliability
  curves.
- Compare raw XGBoost, sigmoid calibration, and isotonic calibration.
- Do not use only AUC to judge probability quality; AUC measures ranking, not
  calibration.

### 4. Statistical Features Not Contributing

The current model appears to ignore the statistical feature block. This matters
for LIME because the final classifier may be driven almost entirely by opaque
embedding dimensions.

Recommended check:

- Train three ablations:
  - semantic-only XGBoost
  - statistical-only XGBoost
  - semantic plus statistical XGBoost
- Compare accuracy, AUC, Brier score, log loss, and calibration curves.
- If semantic-only performs almost identically, the statistical features are not
  helping the current XGBoost model.

## Recommended Plan

### Phase 1: Make The Experiment Auditable

1. Fix the split naming mismatch in `train_xgboost_binary-copy.py`.
2. Save the exact data paths and split names into `training_config.json`.
3. Add probability diagnostics to the training output:
   - Brier score
   - log loss
   - probability quantiles
   - percentage of predictions below `0.01`
   - percentage of predictions above `0.99`
   - percentage in `0.1` to `0.9`
4. Add a reliability curve or calibration table.

Goal: know whether the model is accurate, calibrated, both, or only accurate.

### Phase 2: Test Whether RoBERTa Embeddings Are Too Separable

Run ablation experiments:

1. XGBoost on only the 10 statistical features.
2. XGBoost on only the 1024 RoBERTa semantic features.
3. XGBoost on all 1034 concatenated features.
4. A simple logistic regression on all features.
5. A simple logistic regression on statistical features only.

If the semantic-only model still gives near-99% accuracy and near-99%
probabilities, then the issue is mostly coming from the contrastive embedding
space, not the XGBoost configuration.

### Phase 3: Use A Probability-Friendly Model For LIME

For LIME, consider training a separate auditor-facing model instead of forcing
the production XGBoost model to have softer probabilities.

Good candidates:

- Logistic regression on calibrated features.
- Explainable Boosting Machine if available.
- A shallower XGBoost model with stronger regularization.
- A statistical-feature-only model for text-level interpretability.

The production model can remain optimized for classification accuracy, while
the auditor model can be optimized for stable local explanations.

### Phase 4: Adjust XGBoost If It Must Be The LIME Model

If XGBoost must be used directly for LIME, try making it less saturated:

```python
model_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 100,
    "max_depth": 1,
    "learning_rate": 0.01,
    "min_child_weight": 50,
    "reg_lambda": 50,
    "reg_alpha": 5,
    "subsample": 0.7,
    "colsample_bytree": 0.5,
}
```

This may reduce accuracy slightly, but it can produce less brittle probability
behavior. The tradeoff is worthwhile if the goal is auditing rather than only
classification.

### Phase 5: Improve LIME Setup

LIME perturbs raw text, but the classifier consumes RoBERTa embeddings plus
statistical features. This creates an explanation gap:

- LIME changes words.
- The feature extractor maps the changed text into dense semantic vectors.
- XGBoost splits on anonymous embedding dimensions like `semantic_576`.

Recommended approach:

- Use LIME with the full inference pipeline, not precomputed features.
- Use enough perturbation samples to observe probability movement.
- Audit only essays where the model is not completely saturated, or add an
  abstention/uncertainty band.
- Consider SHAP for XGBoost feature-level explanations and LIME for text-level
  behavioral explanations.

## Suggested Success Criteria

The issue should be considered improved when:

- The model reports calibration metrics, not just accuracy and AUC.
- At least `10%` to `25%` of realistic held-out examples have probabilities in
  a non-saturated range such as `0.05` to `0.95`, or the system explicitly
  documents that the classifier is intentionally high-confidence.
- LIME perturbations cause measurable probability changes around the audited
  essay.
- The final test set is source-disjoint from all model development steps.
- The team knows whether the statistical features add value beyond RoBERTa.

## Practical Recommendation

Do not treat the current `prob_ai` and `prob_human` values as human-readable
confidence yet. Treat them as model scores that are good for classification but
not yet validated for uncertainty.

The best next move is to add calibration diagnostics and run the ablation suite.
If the semantic-only model remains almost perfect, create a separate
auditor-facing model with softer, better-calibrated probabilities for LIME.
That will give clearer explanations without weakening the main detector.
