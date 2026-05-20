These results are excellent. Let me break them down fully.

---

### Overall results

| Metric | Score |
|--------|-------|
| Accuracy | 99.9% |
| AUC | 0.9998 |
| Human F1 | 0.99 |
| AI F1 | 1.00 |

---

### What makes this credible this time

Unlike the previous 1.0 result, this is **not leakage** because:
- Probe trained on 3,899 samples, tested on **held-out 1,672** it never saw
- Checkpoint verified at 499.6MB with correct keys
- `Model loaded cleanly — all keys matched` ✓

---

### The most important number

```
Original AI     detection: 1.0000
Paraphrased AI  detection: 0.9980
```

The gap is only **0.002** — paraphrasing is barely fooling the model at all. This is exactly what your contrastive training was designed to achieve. Pulling label 1 and label 2 together in embedding space worked — the model learned that paraphrased AI text is fundamentally similar to original AI text regardless of surface rewording.

---

### Honest caveats before celebrating

**This is still the val set**, not a truly unseen test set. The model was never trained on val, but the val set came from the same distribution as train. Real-world performance on out-of-distribution text (different AI models, different domains, different paraphrasers) may be lower.

**The val set is imbalanced** — 71% AI vs 29% human. A naive classifier predicting AI always would get 71% accuracy. Your 99.9% is far above that baseline, so the model is genuinely discriminating.

---

### Verdict

**`best_model.pt` is ready for stage 3.** The embeddings are cleanly separable with near-perfect AUC. Whatever classifier you put on top in stage 3 — whether a linear head, MLP, or full fine-tuned classifier — it has excellent embeddings to work with.

What does stage 3 look like?