"""
ci/run_monitoring_gate.py
--------------------------
CI regression gate for monitoring metrics.

Checks:
- p95 latency <= threshold
- Faithfulness score >= threshold
- Error rate <= threshold

Exits with code 1 if any check fails — this fails the GitHub Actions build.

Usage:
    python ci/run_monitoring_gate.py
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.metrics import get_collector

# Thresholds
P95_LATENCY_THRESHOLD    = float(os.getenv("P95_LATENCY_THRESHOLD",    "5.0"))
FAITHFULNESS_THRESHOLD   = float(os.getenv("FAITHFULNESS_THRESHOLD",   "0.80"))
ERROR_RATE_THRESHOLD     = float(os.getenv("ERROR_RATE_THRESHOLD",      "0.05"))


def main():
    print("=" * 60)
    print("  Production RAG — Monitoring Regression Gate")
    print("=" * 60)

    collector = get_collector()
    summary   = collector.summary(window_hours=1.0)

    print(f"\n📊 Metrics (last 1 hour):")
    print(f"  Total queries:  {summary.total_queries}")
    print(f"  Error rate:     {summary.error_rate:.1%}")
    print(f"  Avg latency:    {summary.avg_latency}s")
    print(f"  p50 latency:    {summary.p50_latency}s")
    print(f"  p95 latency:    {summary.p95_latency}s")
    print(f"  p99 latency:    {summary.p99_latency}s")
    print(f"  Total cost:     ${summary.total_cost_usd:.6f}")
    print(f"  Faithfulness:   {summary.avg_faithfulness}")
    print(f"  Relevancy:      {summary.avg_answer_relevancy}")

    if summary.total_queries == 0:
        print("\n⚠️  No queries in the last hour — skipping gate (no data)")
        sys.exit(0)

    checks = collector.check_regression(
        p95_threshold=P95_LATENCY_THRESHOLD,
        faithfulness_threshold=FAITHFULNESS_THRESHOLD,
        error_rate_threshold=ERROR_RATE_THRESHOLD,
    )

    print(f"\n{'=' * 60}")
    print("  Regression Check Results")
    print(f"{'=' * 60}")
    print(f"  {'Check':<25} {'Value':>8}  {'Threshold':>10}  {'Status':>8}")
    print(f"  {'-' * 55}")

    for key in ["p95_latency", "faithfulness", "error_rate"]:
        c = checks[key]
        status = "✅ PASS" if c["passed"] else "❌ FAIL"
        val    = f"{c['value']:.3f}" if c["value"] is not None else "N/A"
        print(f"  {key:<25} {val:>8}  {c['threshold']:>10.3f}  {status:>8}")

    print(f"\n{'=' * 60}")

    if checks["overall_passed"]:
        print("  ✅ All checks passed — CI gate APPROVED")
        sys.exit(0)
    else:
        print("  ❌ CI gate FAILED:")
        for key in ["p95_latency", "faithfulness", "error_rate"]:
            c = checks[key]
            if not c["passed"]:
                print(f"     {key}: {c['value']:.3f} violated threshold {c['threshold']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
