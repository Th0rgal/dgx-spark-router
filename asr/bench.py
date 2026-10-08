#!/usr/bin/env python3
"""Benchmark Cohere Transcribe (NVFP4 vs BF16 reference) on the Spark.

Modes per clip:
  file    : one POST of the whole WAV, wait for JSON.
  stream  : same POST with stream=true; time to first token (TTFT) and total.
  chunks  : Murmure-style: VAD chunks (<=~20 s, cut at pauses) sent as they
            "arrive"; we report the latency of the LAST chunk only, i.e. what
            the user waits after they stop talking.
WER/CER are computed on normalized text (lowercase, no punctuation).
"""
import io, json, re, statistics, sys, time, unicodedata, urllib.request, uuid, wave
import numpy as np

URL = sys.argv[1]            # e.g. http://127.0.0.1:8000
LABEL = sys.argv[2]
REPEAT = int(sys.argv[3]) if len(sys.argv) > 3 else 3
refs = json.load(open("refs.json"))


def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^\w\s']", " ", s).replace("'", " ")
    return s.split()


def edit(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[len(b)]


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    return edit(r, h) / max(1, len(r))


def cer(ref, hyp):
    r, h = " ".join(norm(ref)), " ".join(norm(hyp))
    return edit(list(r), list(h)) / max(1, len(r))


def multipart(wav_bytes, fields):
    b = uuid.uuid4().hex
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    out.write(f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode())
    out.write(wav_bytes)
    out.write(f"\r\n--{b}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={b}"


def post(wav_bytes, lang, stream=False):
    body, ctype = multipart(wav_bytes, {"model": "cohere-transcribe", "language": lang,
                                        **({"stream": "true"} if stream else {})})
    req = urllib.request.Request(URL + "/v1/audio/transcriptions", data=body, headers={"Content-Type": ctype})
    t0 = time.perf_counter(); first = None; text = ""
    with urllib.request.urlopen(req, timeout=300) as r:
        if not stream:
            text = json.loads(r.read())["text"]
        else:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                delta = (d.get("choices") or [{}])[0].get("delta", {}).get("content")
                if delta:
                    first = first or time.perf_counter()
                    text += delta
    t1 = time.perf_counter()
    return text, t1 - t0, (first - t0) if first else None


def read_wav(path):
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16), w.getframerate()


def to_wav(pcm, sr=16000):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
    return b.getvalue()


def vad_chunks(pcm, sr=16000, min_s=7.0, max_s=20.0, pause_s=0.45):
    """Simplified Murmure Chunker: cut in the middle of the quietest >=pause
    stretch after min_s, forced at the quietest 30 ms frame before max_s."""
    fr = sr * 30 // 1000
    rms = np.array([np.sqrt(np.mean(pcm[i:i + fr].astype(np.float32) ** 2)) for i in range(0, len(pcm) - fr, fr)])
    thr = np.percentile(rms, 20) * 1.5 + 1
    cuts, start = [], 0
    while (len(rms) - start) * 0.03 > max_s:
        lo, hi = start + int(min_s / 0.03), start + int(max_s / 0.03)
        quiet = rms[lo:hi] < thr
        best, run, cut = 0, 0, None
        for i, q in enumerate(quiet):
            run = run + 1 if q else 0
            if run * 0.03 >= pause_s and run > best:
                best, cut = run, lo + i - run // 2
        cut = cut if cut is not None else lo + int(np.argmin(rms[lo:hi]))
        cuts.append(cut * fr); start = cut
    edges = [0] + cuts + [len(pcm)]
    return [pcm[a:b] for a, b in zip(edges, edges[1:])]


rows = []
for name in sorted(refs) + ["fr_long"]:
    path = f"audio/{name}.wav"
    pcm, sr = read_wav(path)
    dur = len(pcm) / sr
    lang = name.split("_")[0]
    wav = open(path, "rb").read()
    post(wav, lang)  # warm-up
    tf, ts, ttft = [], [], []
    for _ in range(REPEAT):
        text, t, _ = post(wav, lang); tf.append(t)
        _, t2, f2 = post(wav, lang, stream=True); ts.append(t2); ttft.append(f2 or t2)
    ref = refs.get(name) or " ".join(refs[f"fr_mls_{i}"] for i in (0, 1, 2, 0, 1, 2))
    # chunked: last-chunk latency (what remains after the speaker stops)
    chunks = vad_chunks(pcm)
    ctext, last = [], []
    for _ in range(REPEAT):
        ctext = []
        for c in chunks:
            tx, t, _ = post(to_wav(c), lang); ctext.append(tx)
        last.append(t)
    rows.append(dict(clip=name, lang=lang, dur=round(dur, 1), file=statistics.median(tf),
                     stream_total=statistics.median(ts), ttft=statistics.median(ttft),
                     chunks=len(chunks), last_chunk=statistics.median(last),
                     rtf=statistics.median(tf) / dur, wer=wer(ref, text), cer=cer(ref, text),
                     wer_chunked=wer(ref, " ".join(ctext)), text=text))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items() if k != "text"}), flush=True)

json.dump(rows, open(f"bench_{LABEL}.json", "w"), ensure_ascii=False, indent=1)

# concurrency: 4 parallel 19 s French requests
import concurrent.futures as cf
wav = open("audio/fr_mls_0.wav", "rb").read()
t0 = time.perf_counter()
with cf.ThreadPoolExecutor(4) as ex:
    lat = list(ex.map(lambda _: post(wav, "fr")[1], range(4)))
wall = time.perf_counter() - t0
print(json.dumps({"concurrency4_wall_s": round(wall, 3), "per_req_s": [round(x, 3) for x in lat],
                  "audio_s_per_s": round(4 * 19.3 / wall, 1)}))
