#!/usr/bin/env python3
"""Streaming probe for glm-5.3-flash through the router: TTFT, decode tok/s,
reasoning/content split, tool calls, NVMe bytes read during the request."""
import base64, json, sys, time, urllib.error, urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"


def nvme_read_bytes():
    for line in open("/proc/diskstats"):
        f = line.split()
        if f[2] == "nvme0n1":
            return int(f[5]) * 512
    return 0


def run(label, messages, max_tokens=512, effort="low", tools=None):
    body = {"model": "glm-5.3-flash", "messages": messages, "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True},
            "reasoning_effort": effort, "temperature": 0}
    if tools:
        body["tools"] = tools
    data = json.dumps(body).encode()
    while True:  # the router answers 503 + Retry-After while another model is protected
        try:
            urllib.request.urlopen(urllib.request.Request(
                URL.replace("/v1/chat/completions", "/health")), timeout=10).read()
            req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
            r0, t0 = nvme_read_bytes(), time.perf_counter()
            r = urllib.request.urlopen(req, timeout=3600)
            break
        except urllib.error.HTTPError as e:
            if e.code != 503:
                raise
            time.sleep(min(int(e.headers.get("Retry-After") or 30), 120))
    first = None; reasoning = content = ""; tool_calls = {}; usage = None; finish = None; chunks = 0
    with r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                piece = (delta.get("reasoning_content") or delta.get("reasoning") or "")
                if piece or delta.get("content") or delta.get("tool_calls"):
                    chunks += 1
                    first = first or time.perf_counter()
                reasoning += piece
                content += delta.get("content") or ""
                for tc in delta.get("tool_calls") or []:
                    slot = tool_calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["arguments"] += fn.get("arguments") or ""
                finish = ch.get("finish_reason") or finish
    t1 = time.perf_counter()
    n = (usage or {}).get("completion_tokens") or 0
    res = {"label": label, "ttft_s": round(first - t0, 2) if first else None,
           "total_s": round(t1 - t0, 2), "completion_tokens": n,
           "decode_tok_s": round((n - 1) / (t1 - first), 2) if first and n > 1 else None,
           "prompt_tokens": (usage or {}).get("prompt_tokens"), "sse_chunks": chunks,
           "finish": finish, "nvme_read_gib": round((nvme_read_bytes() - r0) / 2**30, 2),
           "reasoning_chars": len(reasoning), "content": content.strip()[:400],
           "reasoning_head": reasoning.strip()[:200], "tool_calls": list(tool_calls.values())}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    return res


if __name__ == "__main__":
    which = sys.argv[1]
    q = [{"role": "user", "content": "In two sentences, why does an MoE model with 18B active parameters run faster than a dense 320B model?"}]
    if which == "load":
        run("load+first", [{"role": "user", "content": "Reply with the single word READY."}], max_tokens=64)
    elif which == "cold":
        run("cold-prompt", q)
    elif which == "warm":
        run("warm-repeat", q)
    elif which == "tools":
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                  "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
        run("tool-call", [{"role": "user", "content": "What is the weather in Lisbon right now? Use the tool."}], tools=tools)
    elif which in ("image", "image-warm"):
        png = sys.argv[2]
        url = "data:image/png;base64," + base64.b64encode(open(png, "rb").read()).decode()
        run(which, [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": "Describe this image precisely: shapes, colors, positions and any text."}]}],
            max_tokens=400)
    elif which == "reason":
        run("reasoning-high", [{"role": "user", "content": "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much is the ball? Answer with the amount."}],
            max_tokens=1536, effort="high")
