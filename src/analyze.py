"""Compare conformal coverage with and without an English-to-Persian shift.

Calibration and test questions are disjoint in every repeat, so no question is
used in both roles. The same test questions are used for every scenario within a
repeat, which makes the differences between scenarios paired.

Usage:
    python src/analyze.py results/Qwen3-4B-Instruct-2507.jsonl
    python src/analyze.py results/gemma-4-E2B-it.jsonl --min-letter-mass 0.9

--min-letter-mass keeps only pairs where the four option letters hold at least this
share of the next-token probability in both languages.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conformal import (  # noqa: E402
    apply_temperature,
    aps_prediction_sets,
    aps_scores,
    conformal_threshold,
    coverage,
    expected_calibration_error,
    fit_temperature,
    lac_scores,
    mean_set_size,
    prediction_sets,
)

ALPHA = 0.10
REPEATS = 1000
SMALL_CALIBRATION_SIZES = (25, 50, 100, 200)


def load_results(path: Path, min_letter_mass: float = 0.0) -> dict:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    rows = [r for r in rows if min(r["en_letter_mass"], r["fa_letter_mass"]) >= min_letter_mass]
    result = {
        "y": np.array([r["answer"] for r in rows]),
        "category": np.array([r["category"] for r in rows]),
    }
    for language in ("en", "fa"):
        logp = np.array([r[f"{language}_log_probs"] for r in rows])
        logp = logp - logp.max(axis=1, keepdims=True)
        probs = np.exp(logp)
        result[language] = probs / probs.sum(axis=1, keepdims=True)
        result[f"{language}_log"] = logp
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


def aps_comparison(data: dict, repeats: int = REPEATS, seed: int = 0) -> dict:
    """The main scenarios again, using the APS score instead of LAC."""
    y = data["y"]
    half = len(y) // 2
    rng = np.random.default_rng(seed)
    scenarios = {"en->en": ("en", "en"), "en->fa": ("en", "fa"), "fa->fa": ("fa", "fa")}
    collected = {name: {"coverage": [], "size": [], "empty": []} for name in scenarios}
    for _ in range(repeats):
        order = rng.permutation(len(y))
        calibration, test = order[:half], order[half:]
        for name, (fit_language, test_language) in scenarios.items():
            threshold = conformal_threshold(aps_scores(data[fit_language][calibration], y[calibration]), ALPHA)
            sets = aps_prediction_sets(data[test_language][test], threshold)
            collected[name]["coverage"].append(coverage(sets, y[test]))
            collected[name]["size"].append(mean_set_size(sets))
            collected[name]["empty"].append(float((sets.sum(axis=1) == 0).mean()))
    summary = {
        name: {key: float(np.mean(values)) for key, values in result.items()} for name, result in collected.items()
    }
    gaps = np.array(collected["en->en"]["coverage"]) - np.array(collected["en->fa"]["coverage"])
    low, high = np.percentile(gaps, [2.5, 97.5])
    summary["coverage_gap_en_minus_fa"] = {"mean": float(gaps.mean()), "p2.5": float(low), "p97.5": float(high)}
    return summary


def temperature_comparison(data: dict, repeats: int = 300, seed: int = 0) -> dict:
    """LAC and APS after temperature scaling, to remove numerical saturation.

    Each split has three disjoint parts: 25% fits the temperature, 25% calibrates the
    conformal threshold and 50% is the test set. The temperature and the threshold
    both come from the calibration language.
    """
    y = data["y"]
    n = len(y)
    rng = np.random.default_rng(seed)
    methods = {"LAC": (lac_scores, prediction_sets), "APS": (aps_scores, aps_prediction_sets)}
    scenarios = {"en->en": ("en", "en"), "en->fa": ("en", "fa"), "fa->fa": ("fa", "fa")}
    collected = {
        method: {name: {"coverage": [], "size": []} for name in scenarios} for method in methods
    }
    temperatures = {"en": [], "fa": []}
    for _ in range(repeats):
        order = rng.permutation(n)
        fit, calibration, test = order[: n // 4], order[n // 4 : n // 2], order[n // 2 :]
        fitted = {}
        for language in ("en", "fa"):
            fitted[language] = fit_temperature(data[f"{language}_log"][fit], y[fit])
            temperatures[language].append(fitted[language])
        for method, (score, make_sets) in methods.items():
            for name, (source, target) in scenarios.items():
                temperature = fitted[source]
                calibration_probs = apply_temperature(data[f"{source}_log"][calibration], temperature)
                threshold = conformal_threshold(score(calibration_probs, y[calibration]), ALPHA)
                sets = make_sets(apply_temperature(data[f"{target}_log"][test], temperature), threshold)
                collected[method][name]["coverage"].append(coverage(sets, y[test]))
                collected[method][name]["size"].append(mean_set_size(sets))
    summary = {"temperature": {k: float(np.median(v)) for k, v in temperatures.items()}}
    for method, scenario_results in collected.items():
        gaps = np.array(scenario_results["en->en"]["coverage"]) - np.array(scenario_results["en->fa"]["coverage"])
        low, high = np.percentile(gaps, [2.5, 97.5])
        summary[method] = {
            name: {"coverage": float(np.mean(v["coverage"])), "set_size": float(np.mean(v["size"]))}
            for name, v in scenario_results.items()
        }
        summary[method]["coverage_gap_en_minus_fa"] = {
            "mean": float(gaps.mean()),
            "p2.5": float(low),
            "p97.5": float(high),
        }
    return summary


def saturation_checks(data: dict, temperatures: dict) -> dict:
    """Calibration error after scaling, and how many APS scores tie at the threshold."""
    y = data["y"]
    checks = {}
    for language in ("en", "fa"):
        raw_scores = aps_scores(data[language], y)
        scaled = apply_temperature(data[f"{language}_log"], temperatures[language])
        scaled_scores = aps_scores(scaled, y)
        checks[language] = {
            "ece_after_scaling": expected_calibration_error(scaled, y),
            "raw_aps_tied_share": float((np.abs(raw_scores - conformal_threshold(raw_scores, ALPHA)) < 1e-9).mean()),
            "scaled_aps_threshold": conformal_threshold(scaled_scores, ALPHA),
            "scaled_aps_tied_share": float(
                (np.abs(scaled_scores - conformal_threshold(scaled_scores, ALPHA)) < 1e-9).mean()
            ),
        }
    return checks


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
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--min-letter-mass", type=float, default=0.0)
    args = parser.parse_args()
    path = args.results
    data = load_results(path, args.min_letter_mass)
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

    aps = aps_comparison(data)
    summary["aps"] = aps
    print(f"\nAPS score\n{'calibrate->test':<18}{'coverage':>10}{'set size':>10}{'empty':>8}")
    for name in ("en->en", "en->fa", "fa->fa"):
        row = aps[name]
        print(f"{name:<18}{row['coverage']:>10.3f}{row['size']:>10.2f}{row['empty']:>8.3f}")
    gap = aps["coverage_gap_en_minus_fa"]
    print(f"APS coverage gap: {gap['mean']:.3f} (2.5% to 97.5% of splits: {gap['p2.5']:.3f} to {gap['p97.5']:.3f})")

    gaps = coverage_gap(data)
    low, high = np.percentile(gaps, [2.5, 97.5])
    summary["coverage_gap_en_minus_fa"] = {"mean": float(gaps.mean()), "p2.5": float(low), "p97.5": float(high)}
    print(f"\ncoverage gap, en->en minus en->fa: {gaps.mean():.3f} (2.5% to 97.5% of splits: {low:.3f} to {high:.3f})")

    tails = threshold_tail(data)
    summary["minimum_probability_to_keep_option"] = tails
    print(f"minimum probability to keep an option: en {tails['en']:.2e} | fa {tails['fa']:.2e}")

    scaled = temperature_comparison(data)
    summary["after_temperature_scaling"] = scaled
    print(f"\nafter temperature scaling (median T: en {scaled['temperature']['en']:.2f}, fa {scaled['temperature']['fa']:.2f})")
    for method in ("LAC", "APS"):
        print(method)
        for name in ("en->en", "en->fa", "fa->fa"):
            row = scaled[method][name]
            print(f"  {name:<16}coverage {row['coverage']:.3f}   set size {row['set_size']:.2f}")
        gap = scaled[method]["coverage_gap_en_minus_fa"]
        print(f"  coverage gap {gap['mean']:.3f} (2.5% to 97.5% of splits: {gap['p2.5']:.3f} to {gap['p97.5']:.3f})")

    checks = saturation_checks(data, scaled["temperature"])
    summary["saturation_checks"] = checks
    print("\nsaturation checks")
    for language, row in checks.items():
        print(
            f"  {language}: ECE after scaling {row['ece_after_scaling']:.3f}"
            f" | raw APS ties {row['raw_aps_tied_share']:.3f}"
            f" | scaled APS threshold {row['scaled_aps_threshold']:.3f}"
            f" | scaled APS ties {row['scaled_aps_tied_share']:.3f}"
        )

    table = category_table(data)
    summary["by_category"] = table
    print(f"\n{'category':<18}{'n':>6}{'acc en':>8}{'acc fa':>8}{'cov en':>8}{'cov fa':>8}")
    for category, row in table.items():
        print(
            f"{category:<18}{row['n']:>6}{row['accuracy_en']:>8.3f}{row['accuracy_fa']:>8.3f}"
            f"{row['coverage_en']:>8.3f}{row['coverage_fa']:>8.3f}"
        )

    suffix = f".mass{args.min_letter_mass:g}.summary.json" if args.min_letter_mass > 0 else ".summary.json"
    target = path.with_suffix(suffix)
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsaved {target}")


if __name__ == "__main__":
    main()
