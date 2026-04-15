"""Side-by-side comparison of multiple BounceMetrics results."""

import json
from pathlib import Path

from eval.metrics import BounceMetrics


def build_comparison(results: list[BounceMetrics]) -> dict:
    """Build a comparison structure from multiple BounceMetrics.

    Returns a JSON-serializable dict with:
        - summary_table: condensed comparison rows
        - detailed: full metrics per method
        - ranking: methods ranked by F1 score
    """
    summary_rows = []
    detailed = []

    for m in results:
        summary_rows.append({
            "method": m.method,
            "gt_count": m.gt_count,
            "detected": m.detected_count,
            "matched": m.matched,
            "missed": m.missed,
            "false_positives": m.false_positives,
            "recall": round(m.recall, 4),
            "precision": round(m.precision, 4),
            "f1": round(m.f1, 4),
            "error_mean_m": round(m.landing_error_mean, 4),
            "error_median_m": round(m.landing_error_median, 4),
            "error_p95_m": round(m.landing_error_p95, 4),
        })
        detailed.append(m.to_dict())

    # Rank by F1 (descending), break ties by recall
    ranking = sorted(
        [{"method": r["method"], "f1": r["f1"], "recall": r["recall"]}
         for r in summary_rows],
        key=lambda x: (x["f1"], x["recall"]),
        reverse=True,
    )
    for i, r in enumerate(ranking):
        r["rank"] = i + 1

    return {
        "summary_table": summary_rows,
        "ranking": ranking,
        "detailed": detailed,
    }


def save_comparison(results: list[BounceMetrics], output_path: Path) -> dict:
    """Build comparison and save to JSON file.

    Args:
        results: list of BounceMetrics from different methods
        output_path: path to write comparison.json

    Returns:
        The comparison dict that was saved
    """
    comparison = build_comparison(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    return comparison
