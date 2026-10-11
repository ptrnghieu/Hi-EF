# Accesses to the Hi-EF test split

| Access | Models | Status |
|---|---|---|
| 1 | recognise-then-transition and trajectory forecasters, a benchmark-style forecaster (earlier model family) | single preregistered run; invalid for the trajectory arms (caching bug) |
| 2, 3 | same family: re-run after the fix; 5-fold CV over all 53 episodes | announced; execution not verified |
| 4 | PartiSet, the Baseline without body branch and internal variants | single preregistered run (`g10_test_preregistered`) |
| 5 | a class-rebalanced variant of PartiSet | single read after access 4; not reported |
| 6 | seven ViT baselines with body branch | read after access 4; not preregistered (`b2_vit_baselines_body`) |
| 7 | Qwen2.5-VL-7B-Instruct, zero-shot and few-shot | prompt fixed on validation rows (`l1_qwen_vl_zero_few_shot`) |
| 8 | released-code baseline | `b3_released_code_baseline` |

The MELD test split was read by PartiSet and the baselines (the first run produced no usable PartiSet output because three input
values were not finite, and was repeated after setting them to zero), by the body-branch baselines and by the language model.
