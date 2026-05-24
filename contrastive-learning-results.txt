1	Results

This chapter presents the empirical results of the contrastive-learning stage for AI-text detection, using the run logs and evaluation outputs in `Contrastive-Learning/` (especially `cleaned_logs_one.txt` and `evaluation results.txt`). The chapter reports model convergence, held-out detection performance, robustness to paraphrasing, and practical runtime/performance indicators.

1.1	Introduction:

The purpose of this chapter is to answer the central research questions of the study:

- RQ1: Can supervised contrastive learning produce embeddings that clearly separate human-written and AI-generated essays?
- RQ2: Does the model remain effective when AI-generated essays are paraphrased?

The main objective was to train a RoBERTa-based contrastive encoder and evaluate whether the learned embedding space supports robust downstream detection.

To achieve this, the study used:

- Training logs from `stage2_roberta_finetune.py` (`cleaned_logs_one.txt`) to track optimization behavior across epochs.
- Held-out evaluation logs from `evaluate_using_test.py` (`evaluation results.txt`) to measure binary detection quality (Human vs AI) on unseen evaluation data.
- Standard classification metrics and diagnostic indicators, including precision, recall, F1-score, accuracy, AUC, confusion matrix, and subgroup recall (Original AI vs Paraphrased AI).

The contrastive fine-tuning configuration used `roberta-large`, `contrastive_mode="label"`, gradient checkpointing, mixed precision, total steps = 6,935 with 416 warmup steps.

1.2	Presentation of Findings:

1.2.1	Dataset Composition for Held-Out Evaluation

The held-out probe evaluation used separate probe-training and evaluation files:

| Split | Total | Human (0) | Original AI (1) | Paraphrased AI (2) |
|---|---:|---:|---:|---:|
| Probe train (`full_train_pool.csv`) | 44,414 | 12,892 (29.03%) | 8,945 (20.14%) | 22,577 (50.83%) |
| Held-out eval (`test.csv`) | 6,088 | 1,612 (26.48%) | 1,119 (18.38%) | 3,357 (55.14%) |

This distribution shows that paraphrased AI is the largest class in both training and evaluation, which is relevant to the study’s robustness goal.

1.2.2	Contrastive Training Convergence (Stage 2)

The epoch summaries extracted from `cleaned_logs_one.txt` are shown below:

| Epoch | Train Loss | Val Loss | Validation Status |
|---:|---:|---:|---|
| 1 | 2.1769 | 2.7914 | New best model saved |
| 2 | 2.0293 | 2.8011 | No improvement |
| 3 | 1.9735 | 2.7584 | New best model saved (best overall) |
| 4 | 1.9655 | 2.7909 | No improvement |
| 5 | Partial log only | Partial log only | Run log truncated before summary |

Key observations:

- Training loss decreased from 2.1769 (Epoch 1) to 1.9655 (Epoch 4), a 9.71% reduction, indicating stable optimization progress.
- Best validation loss was 2.7584 at Epoch 3, improving by 1.18% from Epoch 1 validation loss.
- Validation loss fluctuations after Epoch 3 suggest diminishing generalization gains with additional epochs.
- Epoch timing (from batch 0 to epoch summary) averaged ~24,266 seconds (~6h44m) per epoch for Epochs 1–4.
- Total time for the first four completed epochs was approximately 26h57m44s.

Important run note:

- `cleaned_logs_one.txt` ends during Epoch 5 (up to batch 400/2775), so no final Epoch 5 summary is available in the captured log.

1.2.3	Held-Out Detection Performance

A logistic regression probe was trained on contrastive embeddings from 44,414 probe-train samples and evaluated on 6,088 held-out samples. The embeddings used in this evaluation had shape:

- Train embedding matrix: (44,414, 1,024)
- Eval embedding matrix: (6,088, 1,024)

Binary classification results (0 = Human, 1 = AI):

| Metric | Human | AI | Overall |
|---|---:|---:|---:|
| Precision | 0.9988 | 0.9996 | - |
| Recall | 0.9988 | 0.9996 | - |
| F1-score | 0.9988 | 0.9996 | - |
| Accuracy | - | - | 0.9993 |
| Macro Avg (P/R/F1) | - | - | 0.9992 |
| Weighted Avg (P/R/F1) | - | - | 0.9993 |
| AUC | - | - | 1.0000 |

Confusion matrix `[[TN, FP], [FN, TP]]`:

- `[[1610, 2], [2, 4474]]`

Derived operational indicators:

- False Positive Rate (Human misclassified as AI): 0.12%
- False Negative Rate (AI missed as Human): 0.045%
- Specificity (Human true negative rate): 99.88%
- Sensitivity/Recall (AI true positive rate): 99.96%

These values show near-perfect separability for the held-out binary task.

1.2.4	Paraphrase Robustness (Crucial Metric)

The most important robustness check compares recall across AI subtypes in the held-out set:

- Original AI recall: 0.9982 (1,119 samples)
- Paraphrased AI recall: 1.0000 (3,357 samples)

Detection-gap analysis:

- Paraphrased recall − Original recall = +0.0018 (0.18 percentage points)

Interpretation:

- The model did not degrade on paraphrased AI in this evaluation; it performed marginally better on paraphrased than original AI.
- This is consistent with the objective of contrastive representation learning for paraphrase resilience.

1.2.5	Computational and Deployment-Relevant Performance

From the same logs:

- Device: CUDA-enabled GPU
- Loaded checkpoint size: 4,268.4 MB
- Probe-train embedding extraction throughput (approx.): 7.38 samples/second
- Held-out embedding extraction throughput (approx.): 8.49 samples/second
- End-to-end extraction throughput over both sets (approx.): 7.49 samples/second

These numbers indicate high-quality performance at the cost of significant model size/compute, which is expected for `roberta-large`.

Reproducibility note:

- The evaluation log indicates the loaded checkpoint was `checkpoint_epoch1.pt` (wrapper reports `epoch=1`, `val_loss=2.7914`) rather than `best_model.pt`. Therefore, the reported held-out metrics are tied to that checkpoint and should be documented exactly as such.

1.3	Conclusion:

The contrastive-learning stage achieved strong empirical performance across optimization and evaluation metrics. Training showed consistent loss reduction and reached its best validation loss at Epoch 3. On held-out evaluation, the embedding space supported near-perfect binary Human-vs-AI detection (Accuracy = 99.93%, AUC = 1.0000), with extremely low false positives and false negatives. Most critically, paraphrased AI detection recall reached 100.00%, indicating that the learned representation is robust against paraphrasing in the evaluated data.

Overall, the results provide strong evidence that the contrastive-learning approach is effective for paraphrase-resistant AI-text detection in this experimental setup.
