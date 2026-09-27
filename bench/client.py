#!/usr/bin/env python3
"""Parallel-request client for the pipeline-parallelism baseline benchmark.

Submits N concurrent /completion requests with a fixed prompt and n_predict,
greedy sampling (temperature 0), and records per-request and aggregate timing.

Usage:
    client.py --n 8 --predict 256 --out results_n8.json [--url http://127.0.0.1:18080]
"""
import argparse
import json
import threading
import time
import urllib.request

WORDS = ("The quick brown fox jumps over the lazy dog while the sun rises over "
         "the quiet valley and the river flows gently beneath the old stone bridge "
         "past fields of wheat and groves of ancient oak trees standing tall. ")


def build_prompt(n_tokens):
    # repeat a fixed sentence to approximate n_tokens (~8 tokens per repetition)
    reps = max(1, n_tokens // 9)
    return (WORDS * reps).strip()


def post_completion(url, prompt, n_predict, seed, results, idx):
    body = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "seed": seed,
        "stream": False,
        "cache_prompt": True,
    }
    req = urllib.request.Request(
        url + "/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            data = json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        results[idx] = {"error": str(e)}
        return
    t1 = time.time()

    timings = data.get("timings", {})
    results[idx] = {
        "wall_s": round(t1 - t0, 4),
        "prompt_n": timings.get("prompt_n", data.get("prompt_eval_count", 0)),
        "predicted_n": timings.get("predicted_n", data.get("eval_count", 0)),
        "prompt_ms": timings.get("prompt_ms"),
        "predicted_ms": timings.get("predicted_ms"),
        "predicted_per_second": timings.get("predicted_per_second"),
        "content": data.get("content", ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18080")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--predict", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    prompt = build_prompt(args.prompt_len)
    results = [None] * args.n
    threads = []
    t0 = time.time()
    for i in range(args.n):
        t = threading.Thread(
            target=post_completion,
            args=(args.url, prompt, args.predict, args.seed, results, i),
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    wall = time.time() - t0

    ok = [r for r in results if r and "error" not in r]
    errs = [r for r in results if r and "error" in r]
    total_predicted = sum(r["predicted_n"] for r in ok)
    total_prompt = sum(r["prompt_n"] for r in ok)

    contents = [r["content"] for r in ok]
    all_identical = bool(contents) and all(c == contents[0] for c in contents)
    all_nonempty = bool(contents) and all(c for c in contents)

    summary = {
        "n": args.n,
        "prompt_len": args.prompt_len,
        "n_predict": args.predict,
        "seed": args.seed,
        "wall_s": round(wall, 4),
        "n_ok": len(ok),
        "n_err": len(errs),
        "total_predicted": total_predicted,
        "total_prompt": total_prompt,
        "aggregate_tokens_per_s": round(total_predicted / wall, 2) if wall > 0 else 0.0,
        "all_outputs_identical": all_identical,
        "all_outputs_nonempty": all_nonempty,
        "errors": [e["error"] for e in errs],
        "per_request": ok,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({
        "n": args.n,
        "wall_s": round(wall, 3),
        "total_predicted": total_predicted,
        "aggregate_tokens_per_s": summary["aggregate_tokens_per_s"],
        "all_outputs_identical": all_identical,
        "all_outputs_nonempty": all_nonempty,
        "n_err": len(errs),
    }, indent=2))


if __name__ == "__main__":
    main()
