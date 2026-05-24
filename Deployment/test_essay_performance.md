# Deployment Performance Report

- Generated (UTC): 2026-05-23T05:56:47.627165+00:00
- Source essay: `/teamspace/studios/this_studio/Robust-Ai-Detector/XGBoost Classifier/test_essay.txt`
- Word count: 571

## Model Outputs
- Production label: **Human-written** (id=0)
- Production P(AI): 0.000000
- Production P(Human): 1.000000
- Auditor P(AI): 0.108108
- Auditor P(Human): 0.891892

## Response Times (ms)
- Normalize + word count: 0.042
- Semantic embedding extraction: 4441.55
- Statistical feature extraction: 2286.754
- Production model inference: 57.938
- Auditor model inference: 6.259
- LIME explanation: 3232.463
- Full pipeline (`analyze_text`, with LIME): 3690.051
- Estimated total (manual path): 10025.007

## LIME Top Tokens
- LIME samples: 8
- LIME features: 12

| Rank | Token | Weight | Signal |
|---:|---|---:|---|
| 1 | better | -0.117472 | Human |
| 2 | was | -0.098107 | Human |
| 3 | pain | -0.095175 | Human |
| 4 | my | -0.095175 | Human |
| 5 | about | -0.095175 | Human |
| 6 | wet | -0.093839 | Human |
| 7 | of | -0.071906 | Human |
| 8 | after | -0.053205 | Human |
| 9 | One | 0.042469 | AI |
| 10 | this | -0.023887 | Human |
| 11 | won't | -0.008118 | Human |
| 12 | he | -0.008118 | Human |
