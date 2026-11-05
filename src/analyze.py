"""Compare conformal coverage with and without an English-to-Persian shift.

Calibration and test questions are disjoint in every repeat, so no question is
used in both roles. The same test questions are used for every scenario within a
repeat, which makes the differences between scenarios paired.

Usage:
    python src/analyze.py results/Qwen3-4B-Instruct-2507-4bit.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conformal import (  # noqa: E402
    conformal_threshold,
    coverage,
    expected_calibration_error,
    lac_scores,
    mean_set_size,
    prediction_sets,
)

ALPHA = 0.10
REPEATS = 1000
SMALL_CALIBRATION_SIZES = (25, 50, 100, 200)


def load_results(path: Path) -> dict:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    result = {
        "y": np.array([r["answer"] for r in rows]),
        "category": np.array([r["category"] for r in rows]),
    }
    for language in ("en", "fa"):
        logp = np.array([r[f"{language}_log_probs"] for r in rows])
        logp = logp - logp.max(axis=1, keepdims=True)
        probs = np.exp(logp)
        result[language] = probs / probs.sum(axis=1, keepdims=True)
        result[f"{language}_mass"] = np.array([r[f"{language}_letter_mass"] for r in rows])
    return result


def describe(data: dict) -> None:
    y = data["y"]
    print(f"pairs: {len(y)}")
    for language in ("en", "fa"):
        probs = data[language]
        correct = probs.argmax(axis=1) == y
        print(
            f"{language}: accuracy {correct.mean():.3f}"
            f" | mean confidence {probs.max(axis=1).mean():.3f}"
            f" | ECE {expected_calibration_error(probs, y):.3f}"
            f" | median letter mass {np.median(data[language + '_mass']):.3f}"
        )
    both = (data["en"].argmax(axis=1) == y) & (data["fa"].argmax(axis=1) == y)
    print(f"correct in both languages: {both.mean():.3f}")


def run_scenarios(data: dict, repeats: int = REPEATS, seed: int = 0) -> dict:
    y = data["y"]
    n = len(y)
    half = n // 2
    rng = np.random.default_rng(seed)
    sizes = [s for s in SMALL_CALIBRATION_SIZES if s < half]
    scenarios = ["en->en", "en->fa", "fa->fa"] + [f"fa({s})->fa" for s in sizes]
    collected = {name: {"coverage": [], "size": []} for name in scenarios}

    for _ in range(repeats):
        order = rng.permutation(n)
        calibration, test = order[:half], order[half:]
        fits = {
            "en->en": ("en", "en", calibration),
            "en->fa": ("en", "fa", calibration),
            "fa->fa": ("fa", "fa", calibration),
        }
        for s in sizes:
            fits[f"fa({s})->fa"] = ("fa", "fa", calibration[:s])
        for name, (fit_language, test_language, indices) in fits.items():
            threshold = conformal_threshold(lac_scores(data[fit_language][indices], y[indices]), ALPHA)
            sets = prediction_sets(data[test_language][test], threshold)
            collected[name]["coverage"].append(coverage(sets, y[test]))
            collected[name]["size"].append(mean_set_size(sets))
    return collected


def coverage_gap(data: dict, repeats: int = REPEATS, seed: int = 0) -> np.ndarray:
    """Paired difference in coverage: English threshold on English minus on Persian."""
    y = data["y"]
    half = len(y) // 2
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(repeats):
        order = rng.permutation(len(y))
        calibration, test = order[:half], order[half:]
        threshold = conformal_threshold(lac_scores(data["en"][calibration], y[calibration]), ALPHA)
        gaps.append(
            coverage(prediction_sets(data["en"][test], threshold), y[test])
            - coverage(prediction_sets(data["fa"][test], threshold), y[test])
        )
    return np.array(gaps)


def threshold_tail(data: dict) -> dict:
    """Probability a kept option must reach, using all questions for calibration.

    An option stays in the set when its probability is at least 1 - threshold.
    """
    y = data["y"]
    return {
        language: 1.0 - conformal_threshold(lac_scores(data[language], y), ALPHA) for language in ("en", "fa")
    }


def category_table(data: dict, repeats: int = 200, seed: int = 0) -> dict:
    """Coverage by subject category, with the threshold always fitted on English."""
    y, categories = data["y"], data["category"]
    half = len(y) // 2
    rng = np.random.default_rng(seed)
    table = {}
    for category in sorted(set(categories.tolist())):
        members = categories == category
        english, persian = [], []
        for _ in range(repeats):
            order = rng.permutation(len(y))
            calibration, test = order[:half], order[half:]
            test = test[members[test]]
            threshold = conformal_threshold(lac_scores(data["en"][calibration], y[calibration]), ALPHA)
            english.append(coverage(prediction_sets(data["en"][test], threshold), y[test]))
            persian.append(coverage(prediction_sets(data["fa"][test], threshold), y[test]))
        table[category] = {
            "n": int(members.sum()),
            "accuracy_en": float((data["en"][members].argmax(axis=1) == y[members]).mean()),
            "accuracy_fa": float((data["fa"][members].argmax(axis=1) == y[members]).mean()),
            "coverage_en": float(np.mean(english)),
            "coverage_fa": float(np.mean(persian)),
        }
    return table


def main() -> None:
    path = Path(sys.argv[1])
    data = load_results(path)
    describe(data)
    print(f"\ntarget coverage {1 - ALPHA:.2f}, {REPEATS} random splits")
    print(f"{'calibrate->test':<18}{'coverage':>10}{'sd':>8}{'p5':>8}{'set size':>10}")
    summary = {}
    for name, values in run_scenarios(data).items():
        cov = np.array(values["coverage"])
        size = np.array(values["size"])
        summary[name] = {
            "coverage": float(cov.mean()),
            "coverage_sd": float(cov.std()),
            "coverage_p5": float(np.percentile(cov, 5)),
            "set_size": float(size.mean()),
        }
        print(f"{name:<18}{cov.mean():>10.3f}{cov.std():>8.3f}{np.percentile(cov, 5):>8.3f}{size.mean():>10.2f}")

    gaps = coverage_gap(data)
    low, high = np.percentile(gaps, [2.5, 97.5])
    summary["coverage_gap_en_minus_fa"] = {"mean": float(gaps.mean()), "p2.5": float(low), "p97.5": float(high)}
    print(f"\ncoverage gap, en->en minus en->fa: {gaps.mean():.3f} (2.5% to 97.5% of splits: {low:.3f} to {high:.3f})")

    tails = threshold_tail(data)
    summary["minimum_probability_to_keep_option"] = tails
    print(f"minimum probability to keep an option: en {tails['en']:.2e} | fa {tails['fa']:.2e}")

    table = category_table(data)
    summary["by_category"] = table
    print(f"\n{'category':<18}{'n':>6}{'acc en':>8}{'acc fa':>8}{'cov en':>8}{'cov fa':>8}")
    for category, row in table.items():
        print(
            f"{category:<18}{row['n']:>6}{row['accuracy_en']:>8.3f}{row['accuracy_fa']:>8.3f}"
            f"{row['coverage_en']:>8.3f}{row['coverage_fa']:>8.3f}"
        )

    target = path.with_suffix(".summary.json")
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsaved {target}")


if __name__ == "__main__":
    main()
