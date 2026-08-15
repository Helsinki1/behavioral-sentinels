# Behavioral Sentinels — Results Summary

200 synthetic state book-keeping tasks, horizons 15-35 turns. Primary prediction window K=5 turns. TP = signal fired at/before the first hallucination and within K turns of it; FP = fired on a clean trajectory; FN = hallucination not predicted (no firing, fired late, or window exceeded); TN = clean and silent. Context-length/turn-number rows use the best-F1 threshold from their sweep (see Traditional/*/summary.md). Random compaction is an analytic expectation.


## model: gpt-4o-mini (proprietary), K=5

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| say_my_name | - | 0.000 | 0.000 | 0 | 0 | 199 | 1 | 0.000 | - |
| remember_fact | - | 0.000 | 0.000 | 0 | 0 | 199 | 1 | 0.000 | - |
| format_response | - | 0.000 | 0.000 | 0 | 0 | 188 | 12 | 0.000 | - |
| variable_check | 1.000 | 0.415 | 0.587 | 83 | 0 | 117 | 0 | 0.690 | 4.000 |
| early_decision | - | 0.000 | 0.000 | 0 | 0 | 199 | 1 | 0.000 | - |
| multi_resolution | 0.857 | 0.030 | 0.058 | 6 | 1 | 193 | 0 | 0.050 | 3 |
| Traditional/context_length (θ=1000) | 0.889 | 0.040 | 0.077 | 8 | 1 | 190 | 1 | 0.050 | 3 |
| Traditional/turn_number (θ=5) | 0.981 | 0.530 | 0.689 | 105 | 2 | 93 | 0 | 0.790 | 4.000 |
| Traditional/LLM_judge | 0.979 | 0.470 | 0.635 | 93 | 2 | 105 | 0 | 0.985 | 6 |
| Traditional/random_compaction | 0.957 | 0.227 | 0.367 | 44.920 | 2.000 | 153.080 | 0.000 | 1.000 | 4.000 |

Hallucination base rate (canary runs vary slightly): 0.995

## model: gpt-oss-20b (open), K=5

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| say_my_name | 1.000 | 0.006 | 0.012 | 1 | 0 | 162 | 37 | 0.005 | 0 |
| remember_fact | 1.000 | 0.024 | 0.047 | 4 | 0 | 161 | 35 | 0.020 | 0.000 |
| format_response | 1.000 | 0.014 | 0.027 | 2 | 0 | 145 | 53 | 0.010 | 0.000 |
| variable_check | 0.629 | 0.227 | 0.333 | 39 | 23 | 133 | 5 | 0.595 | 7.000 |
| early_decision | 1.000 | 0.012 | 0.024 | 2 | 0 | 161 | 37 | 0.010 | 0.000 |
| multi_resolution | 0.750 | 0.037 | 0.070 | 6 | 2 | 158 | 34 | 0.050 | 0.500 |
| Traditional/context_length (θ=1000) | 0.524 | 0.362 | 0.429 | 54 | 49 | 95 | 2 | 0.705 | 4.000 |
| Traditional/turn_number (θ=15) | 0.500 | 0.342 | 0.406 | 51 | 51 | 98 | 0 | 0.690 | 5 |
| Traditional/LLM_judge | 0.338 | 0.174 | 0.230 | 26 | 51 | 123 | 0 | 1.000 | 14 |
| Traditional/random_compaction | 0.401 | 0.229 | 0.291 | 34.120 | 51.000 | 114.880 | 0.000 | 1.000 | 8 |

Hallucination base rate (canary runs vary slightly): 0.815
