# Recognize-then-Transition (RtT) gates

Hypothesis: on Hi-EF, forecasting B's emotion splits into (a) recognizing A's state in clip III and
(b) an A→B transition. Part (b) is nearly solved by a 7×7 table: with gold A emotion, `P(B | E_A)`
reaches 28.10 UAR / 39.72 WAR on the locked validation split, above every deployable model so far.
Part (a) is the bottleneck, so these gates test ways to recognize A better.

| File | What it does |
|---|---|
| `cheap_baselines.py` | Majority, Copy-A, Markov (gold A) and TF-IDF baselines, train → val, source-bootstrap CIs |
| `g1_llm_recognition.ipynb` | G1: an OpenAI LLM reads subtitle lines I–III, predicts A's emotion/polarity and B directly; A posterior → `P(B \| E_A)` |
| `g2_recognizer_all_labels.ipynb` | G2: multimodal A recognizer trained on clip III only vs. all labeled clips (III ∪ IV), 5 seeds; A posterior → `P(B \| E_A)` |
| `build_notebooks.py` | Regenerates both notebooks |

All three use the locked split `source_folder_split_seed42.csv` (1,993 / 428 / 409 MCIS, 37 / 8 / 8 episodes).
The test split raises an error unless `UNLOCK_TEST = True`.

## Running on Kaggle

1. Attach `hi-ef-dataset`, `hi-ef-features-v2` and a dataset holding `source_folder_split_seed42.csv`;
   fix the paths in each notebook's CONFIG cell if they differ.
2. G1 only: add the Kaggle secret `OPENAI_API_KEY` and set `MODEL`. Responses are cached in `/kaggle/working`.
3. Run G2 first if you want G1's optional ensemble cell to pick up `g2_val_probs_ALL.npz`.

## Gate criteria

* G1: LLM `emotion_A` UAR above the frozen recognizer (23.21).
* G2: arm `ALL` beats arm `III` on A recognition UAR, across seeds.
* Either gate passing → the RtT forecast should move toward the oracle (≈28 UAR / ≈40 WAR);
  it must beat B1 (24.65 / 35.79) with a source-bootstrap CI that excludes zero before the test split is opened.
