"""
API Latency Benchmark.

Measures actual inference latency: avg, p50, p95, throughput.

Usage:
    python scripts/benchmark_api.py [--url http://localhost:8000] [--n 100]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_benchmark(
    base_url: str = "http://localhost:8000",
    n_requests: int = 100,
    user_id: str = "U10000",
    top_k: int = 10,
) -> dict[str, float]:
    """
    Benchmark the /recommend endpoint.

    Returns actual measured latency statistics.
    """
    print(f"\nBenchmarking {base_url}/recommend")
    print(f"  Requests: {n_requests}")
    print(f"  User: {user_id}, top_k: {top_k}")
    print("-" * 50)

    # Warm-up
    for _ in range(3):
        try:
            requests.post(
                f"{base_url}/recommend",
                json={"user_id": user_id, "top_k": top_k},
                timeout=10,
            )
        except Exception:
            pass

    # Benchmark
    latencies = []
    errors = 0

    for i in range(n_requests):
        start = time.perf_counter()
        try:
            resp = requests.post(
                f"{base_url}/recommend",
                json={"user_id": user_id, "top_k": top_k},
                timeout=10,
            )
            elapsed = (time.perf_counter() - start) * 1000  # ms
            if resp.status_code == 200:
                latencies.append(elapsed)
            else:
                errors += 1
        except Exception:
            errors += 1

    if not latencies:
        print("ERROR: No successful requests!")
        return {}

    latencies.sort()
    results = {
        "n_requests": n_requests,
        "successful": len(latencies),
        "errors": errors,
        "avg_ms": statistics.mean(latencies),
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "p99_ms": latencies[int(len(latencies) * 0.99)] if len(latencies) > 10 else latencies[-1],
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "throughput_rps": len(latencies) / (sum(latencies) / 1000),
    }

    print("\nRESULTS")
    print("=" * 50)
    print(f"  Successful: {results['successful']}/{n_requests}")
    print(f"  Average:    {results['avg_ms']:.1f} ms")
    print(f"  P50:        {results['p50_ms']:.1f} ms")
    print(f"  P95:        {results['p95_ms']:.1f} ms")
    print(f"  P99:        {results['p99_ms']:.1f} ms")
    print(f"  Min:        {results['min_ms']:.1f} ms")
    print(f"  Max:        {results['max_ms']:.1f} ms")
    print(f"  Throughput: {results['throughput_rps']:.1f} req/s")
    print("=" * 50)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the recommendation API")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--user-id", default="U10000")
    args = parser.parse_args()

    results = run_benchmark(args.url, args.n, args.user_id)

    if results:
        out_path = Path("docs/figures/benchmark_results.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")
