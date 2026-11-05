# Conformal Language Shift

Does a conformal prediction guarantee survive translation? This project calibrates a language model on English multiple-choice questions and tests it on the same kind of questions in Persian.

**Status: work in progress.** The results below are preliminary and come from one model and 2,000 question pairs.

## Method

- **Data:** paired English and Persian questions from [Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU).
- **Model:** [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), run in `float16` on a Colab T4 GPU.
- **Scores:** the probabilities of options A, B, C and D at the first generated token, normalized over the four options.
- **Conformal method:** split conformal prediction with the LAC score, targeting 90% coverage.
- **Evaluation:** 1,000 random splits. Calibration and test questions never overlap.

## Preliminary results

| Language | Accuracy | Mean confidence | ECE |
|---|---:|---:|---:|
| English | 0.701 | 0.965 | 0.265 |
| Persian | 0.496 | 0.942 | 0.446 |

| Calibrate → test | Coverage | Mean set size (out of 4) |
|---|---:|---:|
| English → English | 0.900 | 1.99 |
| English → Persian | 0.892 | 2.83 |
| Persian → Persian | 0.900 | 2.88 |

- **Coverage roughly held.** The gap between English → English and English → Persian averaged 0.8 percentage points. Across splits, 95% of the gaps fell between -0.8 and 3.3 points. The expected large drop did not appear.
- **The cost appeared in set size.** Prediction sets grew from 1.99 to 2.83 options, which makes them much less informative. Persian accuracy was also 20.5 points lower.
- **Coverage varied by subject.** With the English threshold, Persian coverage ranged from 0.854 in Humanities to 0.953 in the Other category. Some groups are small, and this needs testing on more data before drawing conclusions.

## Limitations

- The results come from a single model, so they may not generalize.
- The model is badly overconfident. An option stays in the set if its probability is above about 3 × 10⁻⁷, so the sets depend on very small probabilities. These values may be sensitive to `float16` rounding.
- The Persian questions are translations, not questions written natively in Persian.

## Next steps

- Repeat the experiment with a second model and more question pairs.
- Add the APS score to test whether the result depends on the score function. `aps_scores` in `src/conformal.py` is not implemented yet.
- Test whether the small probabilities are stable under higher numerical precision.

## Reproduce

1. Open `colab_score.ipynb` in Google Colab, choose a T4 GPU and run all cells.
2. Put the output file in `results/`.
3. Run the analysis and the tests:

```bash
pip install -r requirements.txt
python src/analyze.py results/Qwen3-4B-Instruct-2507.jsonl
pytest tests
```

The results used here are in `results/`, so you can run the analysis without a GPU.

## Files

| Path | Purpose |
|---|---|
| `colab_score.ipynb` | Runs the model and saves the option probabilities |
| `src/conformal.py` | Conformal prediction and calibration metrics |
| `src/analyze.py` | Produces the results above |
| `tests/test_conformal.py` | Tests for the conformal code on synthetic data |
| `results/` | Model scores and the result summary |
