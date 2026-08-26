# Deployment paper post-analysis

This layer is provider-free and must run only after both fail-closed source analyses finish. It reconstructs all paper statistics from their row-level source-task records; upstream summary arrays are checked only as integrity cross-checks.

## 1. Produce the source analyses after exact run completion

```bash
python3 -m experiments12.adaptive_analysis12 extract --run-id e12-deploy-online-evolving-luna-40-v1 --manifest-sha256 7294e3edcd3468aeb8881182f77a3f22c9d9de41dff1dba8f183e9fe1753c9a7 --output experiments12/artifacts/e12-deploy-online-evolving-luna-40-v1/results/adaptive-analysis.json --figures experiments12/artifacts/e12-deploy-online-evolving-luna-40-v1/results/adaptive-figures --artifacts experiments12/artifacts --bootstrap-iterations 2000 --bootstrap-seed 12012

python3 -m experiments12.two_pass_analysis12 validate --run-id e12-deploy-twopass-yoked-evolving-luna-40-v1 --manifest-sha256 8cd3194cb460aa5e77a8d9b7b7aeee01f6c81df7d655f22f7a03a529f4062250 --output experiments12/artifacts/e12-deploy-twopass-yoked-evolving-luna-40-v1/results/validation-two-pass.json --artifacts experiments12/artifacts

python3 -m experiments12.two_pass_analysis12 extract --run-id e12-deploy-twopass-yoked-evolving-luna-40-v1 --manifest-sha256 8cd3194cb460aa5e77a8d9b7b7aeee01f6c81df7d655f22f7a03a529f4062250 --output experiments12/artifacts/e12-deploy-twopass-yoked-evolving-luna-40-v1/results/two-pass-analysis.json --tables experiments12/artifacts/e12-deploy-twopass-yoked-evolving-luna-40-v1/results/two-pass-tables --figures experiments12/artifacts/e12-deploy-twopass-yoked-evolving-luna-40-v1/results/two-pass-figures --artifacts experiments12/artifacts --bootstrap-iterations 2000 --bootstrap-seed 12012
```

## 2. Run this post-analysis

```bash
python3 experiments12/generated/deployment_post_analysis12.py --online experiments12/artifacts/e12-deploy-online-evolving-luna-40-v1/results/adaptive-analysis.json --yoked experiments12/artifacts/e12-deploy-twopass-yoked-evolving-luna-40-v1/results/two-pass-analysis.json --output-dir experiments12/generated/deployment-paper-post-analysis-v1 --bootstrap-iterations 2000
```

Static readiness check, which does not load either analysis or write outputs:

```bash
python3 experiments12/generated/deployment_post_analysis12.py --dry-check
```

## Outputs

- `deployment-paper-post-analysis.json`: complete exact results and provenance.
- Nine CSVs: primary online summaries/effects/interactions and yoked sensitivity summaries/effects/interactions.
- Five polished SVGs with exact `.data.json` sidecars:
  - grouped online success under the three paper classes and four operators;
  - actual task-level firing/action incidence;
  - end-to-end online observer tokens, total tokens, and cost;
  - paired online method-by-operator success interactions;
  - controlled yoked success sensitivity.

The online results are the primary ecological deployment estimates. The yoked results are explicitly labeled as a controlled sensitivity on the same 40 source tasks, with one active-anchored action at checkpoint 1 in every cell; they are neither natural trigger-timing evidence nor an independent replication. Online resources are end-to-end, while yoked resources exclude the frozen pass-one passive-observer cost.
