# Retrained sham-listener control (development CV, 10 seeds, 45 episodes)

The listener slot of every sample is replaced by the listener slot of another, randomly chosen sample (a fixed derangement of the table), in training and in evaluation. The model, the features, the number of tokens and the protocol are those of the full model; only the content of the listener slot changes. `Full` reproduces the earlier ten-seed run exactly (maximum absolute difference of the probabilities 0.0).

| Arm | Ensemble UAR | Ensemble WAR | Per-seed UAR (mean ± SD) |
|---|---|---|---|
| Full PartiSet (own listener) | 26.24 | 38.04 | 24.74 ± 0.97 |
| Listener slot emptied (no listener) | 24.51 | 35.98 | 23.12 ± 0.66 |
| Sham listener (listener of another sample) | 23.36 | 33.62 | 22.51 ± 0.88 |

Paired differences, 95% interval from resampling whole episodes (a − b):

| a − b | ΔUAR [95% CI] | ΔWAR [95% CI] | ep+ (UAR) |
|---|---|---|---|
| Full − sham listener | +2.88 [1.39, 4.22] | +4.42 [2.39, 6.38] | 34/45 |
| Full − no listener | +1.73 [0.67, 2.84] | +2.07 [0.55, 3.69] | 25/45 |
| Sham listener − no listener | −1.15 [−2.19, −0.06] | −2.35 [−4.13, −0.56] | 14/45 |

`ep+` is the number of episodes in which `a` has the higher UAR. The comparisons are not corrected for multiple comparisons.
