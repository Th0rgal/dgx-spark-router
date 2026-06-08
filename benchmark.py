#!/usr/bin/env python3
"""Compare router-served models on a fixed prompt suite.

Drives the router (which swaps backends on demand), records latency and
throughput, and saves both raw JSON and a markdown summary. Stdlib only.

Usage:
  python3 benchmark.py --models nemotron,qwen3.6,gemma-4,gpt-oss
  python3 benchmark.py --models fast --prompts speed
"""
import argparse, json, time, urllib.request, urllib.error, sys, statistics
from datetime import datetime, timezone

ROUTER = "http://localhost:8000/v1/chat/completions"

# (id, category, system, user, max_tokens)
PROMPTS = [
    ("reasoning", "reasoning", None,
     "Three people check into a hotel room that costs $30. They each pay $10. "
     "Later the clerk realizes the room was only $25 and sends $5 back via the bellhop, "
     "who keeps $2 and gives each guest $1. Now each guest paid $9 (total $27) plus the "
     "bellhop's $2 is $29. Where is the missing dollar? Explain the flaw precisely.", 700),
    ("math", "math", None,
     "Let f(x)=x^3-3x+1. Find the number of real roots in (0,1) and justify rigorously "
     "using the intermediate value theorem and monotonicity. Give exact reasoning.", 700),
    ("code", "code", "You are a precise senior engineer. Output only code unless asked.",
     "Write a Python function `merge_intervals(intervals)` that merges overlapping closed "
     "intervals and returns them sorted. Handle empty input and single intervals. Include 3 "
     "doctest examples covering an edge case.", 700),
    ("tooljson", "instruction", None,
     "Extract the fields and return STRICT minified JSON with keys name, age, city and nothing "
     "else: 'Hi, I'm Marie, I'm 34 and I live in Lyon.' Output only the JSON.", 200),
    ("french", "multilingual", None,
     "Explique en francais, en 4 phrases maximum et pour un lyceen, ce qu'est l'intrication "
     "quantique et pourquoi elle n'autorise pas la communication plus rapide que la lumiere.", 400),
    ("science", "knowledge", None,
     "Concisely explain why ice floats on water at the molecular level, and one consequence "
     "for aquatic life in winter. 5 sentences max.", 350),
    ("speed", "latency", None, "Reply with exactly the single word: pong", 16),
]


def chat(model, system, user, max_tokens, timeout):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    body = json.dumps({
        "model": model, "messages": msgs,
        "max_tokens": max_tokens, "temperature": 0.2, "stream": False,
    }).encode()
    req = urllib.request.Request(ROUTER, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    elapsed = time.time() - t0
    msg = d.get("choices", [{}])[0].get("message", {})
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    usage = d.get("usage", {})
    ctoks = usage.get("completion_tokens")
    return {
        "elapsed_s": round(elapsed, 2),
        "completion_tokens": ctoks,
        "tok_per_s": round(ctoks / elapsed, 1) if ctoks and elapsed else None,
        "content": content,
        "reasoning_chars": len(reasoning),
        "content_chars": len(content),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated router model names")
    ap.add_argument("--prompts", default="all", help="comma-separated prompt ids or 'all'")
    ap.add_argument("--out", default="bench-results.json")
    ap.add_argument("--swap-timeout", type=int, default=2000, help="cold-start timeout (s)")
    ap.add_argument("--gen-timeout", type=int, default=600)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    want = None if args.prompts == "all" else set(args.prompts.split(","))
    suite = [p for p in PROMPTS if want is None or p[0] in want]

    results = {"started": datetime.now(timezone.utc).isoformat(), "models": {}}
    for model in models:
        print(f"\n=== {model} ===", flush=True)
        # Warmup request triggers the swap/cold-start; allow a long timeout.
        try:
            t0 = time.time()
            chat(model, None, "Reply with: ready", 8, args.swap_timeout)
            print(f"  warmup/swap ok in {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(f"  WARMUP FAILED: {e}", flush=True)
            results["models"][model] = {"error": f"warmup: {e}"}
            continue

        per = {}
        for pid, cat, system, user, mx in suite:
            try:
                r = chat(model, system, user, mx, args.gen_timeout)
                per[pid] = {"category": cat, **r}
                print(f"  {pid:10s} {r['elapsed_s']:6.2f}s  "
                      f"{str(r['tok_per_s']):>6} tok/s  ctoks={r['completion_tokens']}", flush=True)
            except Exception as e:
                per[pid] = {"category": cat, "error": str(e)}
                print(f"  {pid:10s} ERROR {e}", flush=True)
        toks = [v["tok_per_s"] for v in per.values() if v.get("tok_per_s")]
        results["models"][model] = {
            "median_tok_per_s": round(statistics.median(toks), 1) if toks else None,
            "prompts": per,
        }
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # Markdown speed summary
    print("\n\n## Throughput summary (median tok/s)\n")
    print("| model | median tok/s | speed prompt s |")
    print("|---|---|---|")
    for m, d in results["models"].items():
        if "error" in d:
            print(f"| {m} | FAILED | - |")
            continue
        sp = d["prompts"].get("speed", {}).get("elapsed_s", "-")
        print(f"| {m} | {d.get('median_tok_per_s')} | {sp} |")
    print(f"\nRaw results -> {args.out}")


if __name__ == "__main__":
    main()
