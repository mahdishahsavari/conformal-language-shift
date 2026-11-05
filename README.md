# Conformal Language Shift

Does a conformal prediction guarantee survive translation? This project calibrates language models on English multiple-choice questions and tests them on the same questions in Persian.

**Status: preliminary.** The results come from two models and 2,000 question pairs.

## Main finding so far

A threshold calibrated in English did not reliably keep its 90% coverage target on Persian. On pairs where both models answered with an option letter:

- **APS** fell below the target on both models (0.862 and 0.853).
- **LAC** depended on the model: Qwen3 stayed on target (0.910), but Gemma 4 fell to 0.829.
- Accuracy was lower in Persian for both models, and Persian prediction sets were larger than English ones.

## Method

- **Data:** the first 2,000 question pairs, in a fixed random order, from [Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU).
- **Models:** [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) in `float16` and [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) in `bfloat16`, both on a Colab T4 GPU.
- **Scores:** the log-probabilities of options A, B, C and D at the first generated token, normalized over the four options.
- **Letter mass:** the share of next-token probability held by those four letters. A low value means the model wanted to start its answer differently.
- **Conformal methods:** split conformal prediction with the LAC and APS scores, targeting 90% coverage. APS is deterministic, so a set can be empty.
- **Temperature scaling:** fitted by grid search on a separate part of the calibration language.
- **Evaluation:** 300 random splits. Each uses 25% of the questions for the temperature, 25% for the threshold and 50% for testing, with no overlap.

## Preliminary results

### The models (all 2,000 pairs)

| Model | Language | Accuracy | ECE | Fitted temperature | Letter mass below 0.9 |
|---|---|---:|---:|---:|---:|
| Qwen3-4B | English | 0.701 | 0.265 | 7.67 | 1.6% |
| Qwen3-4B | Persian | 0.496 | 0.446 | 10.38 | 1.7% |
| Gemma 4 E2B | English | 0.545 | 0.389 | 6.27 | 13.8% |
| Gemma 4 E2B | Persian | 0.427 | 0.495 | 9.38 | 0.2% |

### Coverage after temperature scaling, pairs with letter mass of at least 0.9 in both languages

This keeps 1,952 pairs for Qwen3-4B and 1,724 for Gemma 4 E2B. The target is 0.90. Set size is out of 4 options.

| Model | Calibrate → test | LAC coverage | LAC set size | APS coverage | APS set size |
|---|---|---:|---:|---:|---:|
| Qwen3-4B | English → English | 0.903 | 1.99 | 0.904 | 2.31 |
| Qwen3-4B | English → Persian | 0.910 | 2.90 | **0.862** | 2.88 |
| Gemma 4 E2B | English → English | 0.899 | 2.33 | 0.901 | 2.81 |
| Gemma 4 E2B | English → Persian | **0.829** | 2.56 | **0.853** | 2.95 |

The coverage gap is English → English minus English → Persian. The range covers 95% of the splits.

| Model | LAC gap (points) | APS gap (points) |
|---|---:|---:|
| Qwen3-4B | -0.7 (-3.4 to 5.4) | 4.2 (2.2 to 6.4) |
| Gemma 4 E2B | 7.0 (5.2 to 8.8) | 4.8 (2.1 to 7.0) |

### Effect of the letter mass filter

On all 2,000 pairs, LAC gives the same picture: English → Persian coverage is 0.904 for Qwen3-4B and 0.839 for Gemma 4 E2B. APS changes for Gemma 4 E2B: coverage is 0.888 and the gap is 2.6 points (-0.2 to 4.8). This number is less trustworthy, because Gemma 4's English APS threshold reaches 1.0 and 11% of scores tie at it without the filter. With the filter, that threshold is 0.964 and ties are below 0.1%.

### Persian → Persian APS

Persian → Persian APS should not be read as a good result. Its threshold reaches 1.0 in both models, 13% to 15% of its scores tie there, and it covers about 0.97 by keeping most options. Better calibration of Persian scores is needed before comparing it.

### Calibrating with a few Persian questions

Without temperature scaling, LAC calibrated on only 100 Persian questions reached average coverage of 0.902 for Qwen3-4B and 0.901 for Gemma 4 E2B. Individual runs vary more: the 5th percentile of coverage was about 0.85. This restores coverage but does not make the sets smaller: mean set size was 2.89 and 3.09.

## Limitations

- The letter mass filter was chosen after seeing the Gemma 4 results, so filtered results are exploratory.
- Why Gemma 4 often avoids a letter in English has not been investigated.
- There are only two models, and they ran at different numerical precisions (`float16` and `bfloat16`).
- Both are small models, so larger models may behave differently.
- The Persian questions are translations, not questions written natively in Persian.
- Why LAC and APS behave differently has not been tested.

## Reproduce

1. Open `colab_score.ipynb` in Google Colab, choose a T4 GPU and run all cells. The notebook is set to Gemma 4 E2B, and a comment in its first cell gives the Qwen3-4B settings.
2. Put the output file in `results/`.
3. Run the analysis and the tests:

```bash
pip install -r requirements.txt
python src/analyze.py results/gemma-4-E2B-it.jsonl
python src/analyze.py results/gemma-4-E2B-it.jsonl --min-letter-mass 0.9
pytest tests
```

The scores for both models are in `results/`, so you can run the analysis without a GPU.

## Files

| Path | Purpose |
|---|---|
| `colab_score.ipynb` | Runs a model and saves the option scores |
| `src/conformal.py` | Conformal prediction, temperature scaling and calibration metrics |
| `src/analyze.py` | Produces the results above |
| `tests/test_conformal.py` | Tests for the conformal code on synthetic data |
| `results/` | Model scores and result summaries |
