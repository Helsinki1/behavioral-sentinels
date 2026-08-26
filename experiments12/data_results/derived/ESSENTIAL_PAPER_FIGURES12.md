# Experiment 12: essential paper figures

All figures below are provider-free derivatives of frozen Experiment 12 results. The three composite SVGs were independently reconstructed from raw rows, visually inspected at full render, and audited for two-column `figure*` use (minimum text size: 7.151 pt at 7-inch width).

## Main paper

1. **Deployment interactions — primary result**  
   `deployment-interaction-confirmatory-v1.svg`  
   SHA256: `d47dabee6bd291ab6aeffe41cdbe227c3ef8fd4b9f5cc5b5a55ac5bc1e8501c4`  
   Use full width. It keeps the natural online policy (n=40) and aggressive checkpoint-1 yoked control (n=40) in separate panels and annotates natural action rates. Four online effects exclude zero: clock + lossy compaction −67.5 pp; passive quiz + lossy compaction −30.0 pp; context use + lossy compaction −20.0 pp; active recomputation + public-state regrounding +12.5 pp. The cumulative n=38 sensitivity preserves every CI classification; one includes-zero point estimate changes from exactly 0 to positive.

2. **Carried active probes and observer effect — central mechanism result**  
   `active-probe-ladder-confirmatory-v1.svg`  
   SHA256: `958448b050709abacc3cb2db590a7691ff4bd859e7bdbcbc27643cc936f61b73`  
   Use full width. The exploratory chore ladder is visibly separated from the powered recomputation forest. Powered recomputation is harmful in 6/7 model-by-benchmark strata, with 3/7 paired intervals strictly below zero; the exploratory burden ladder is non-monotone and must not be described as a dose-response result.

3. **What observation costs — implementation tradeoff**  
   `observer-overhead-confirmatory-v1.svg`  
   SHA256: `78b12d9f0601ec0d22a3582d85932b0c92f8c6cd8c0885e2138833687e545bc6`  
   Use full width. This compares provider/API token, latency, and dollar overhead for active and passive methods. Deterministic local monitor runtime was not measured and is labeled accordingly.

## Signal-quality small multiples

Use these seven precision-recall panels as one 4+3 grid or move the full set to the appendix. They show the central qualification: active recomputation has the highest AUPRC in 3/7 powered slices; passive/context methods lead 4/7, so no class wins universally.

- `../artifacts/e12-confirmatory-evolving-core-v2/results/signal-figures/signal-pr-evolving_intent_gsm8k-deepseek-v4-flash-0731.svg` — `3341cc6f03eab6f4cc2840372f89167d794024f6f1311d7334cec573d145b573`
- `../artifacts/e12-confirmatory-evolving-core-v2/results/signal-figures/signal-pr-evolving_intent_gsm8k-gpt-5.6-luna.svg` — `19e7949be3408823e8b23c8b1cffb92aee4f5537ad3ea79abb2f6b4c1415a04c`
- `../artifacts/e12-confirmatory-evolving-core-v2/results/signal-figures/signal-pr-evolving_intent_gsm8k-gpt-5.6-terra.svg` — `4db9b312062cd7e8cb5c1853f763d5f0e172c2e0ab0046e160c7c510ea305872`
- `../artifacts/e12-confirmatory-evolving-core-v2/results/signal-figures/signal-pr-evolving_intent_gsm8k-gpt-oss-120b.svg` — `e5793ecfc737e0c3e34c7688e931ae011a3c56022a82454d1d67149b33fb8e34`
- `../artifacts/e12-confirmatory-bfcl-core-v3/results/signal-figures/signal-pr-bfcl_multi_turn-gpt-5.6-luna.svg` — `53d08f32e0c6516ef8b83127abcbd5417cf72a870371201c65a4fea28743b0bd`
- `../artifacts/e12-confirmatory-bfcl-core-v3/results/signal-figures/signal-pr-bfcl_multi_turn-gpt-5.6-terra.svg` — `6907e6e7ddacf9abb0f2993637b69f492650bc77a2ae79c85231f03916f602d3`
- `../artifacts/e12-confirmatory-bfcl-core-v3/results/signal-figures/signal-pr-bfcl_multi_turn-gpt-oss-120b.svg` — `47bd4a5ab3be8727c638bfe1138f670e70749a57c413a817777513634b882d05`

## Useful appendix figure

- Absolute online success bars: `adaptive-analysis-staging-v1/analysis/figures/deployment-evolving_intent_gsm8k-gpt-5.6-luna.svg`. Pair this with the main interaction figure; it shows absolute success, while the interaction figure shows paired operator-minus-control effects.

## Required interpretation

- Active means the probe exchange is carried in target history; passive means the observer reads a frozen prefix and never enters target history.
- Evolving Intent provides final-task labels; BFCL provides genuine action traces and turn-level failure opportunities.
- Online deployment covers GPT-5.6 Luna on Evolving Intent only. Natural firing/action rates are unequal and reach 100% for some methods.
- The feedback arm is deterministic quote-only WATCH feedback, not an LLM-generated good/bad critique.
- Claims are trends and setting-specific interactions, not universal rules.

