#!/usr/bin/env python3
"""Multi-Model OpenAI-Compatible Router for DGX Spark"""
import contextlib, hmac, ipaddress, os, json, subprocess, tempfile, time, threading, uuid
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default as _email_default_policy
from pathlib import Path
from gpu_idle import Foreground, router_heartbeat
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import urllib.request, urllib.error

BACKEND_PORT = 8001
# Speech-to-text backend (Cohere Transcribe NVFP4, container vllm-asr). It runs
# beside the chat model, so audio requests never trigger a model swap. They take
# the shared GPU foreground lock like chat requests (pauses vanity, not qwen).
ASR_PORT = 8002
ASR_MODEL = "cohere-transcribe"
ASR_MAX_CONCURRENCY = int(os.environ.get("ROUTER_ASR_MAX_CONCURRENCY", "2"))
ASR_MAX_RETRIES = int(os.environ.get("ROUTER_ASR_MAX_RETRIES", "3"))
ASR_HEALTH_WAIT_SEC = int(os.environ.get("ROUTER_ASR_HEALTH_WAIT_SEC", "150"))
ASR_SEMAPHORE = threading.BoundedSemaphore(max(1, ASR_MAX_CONCURRENCY))
MEMORY_GATEWAY_URL = os.environ.get("SPARK_MEMORY_GATEWAY_URL", "http://127.0.0.1:8890").rstrip("/")
MEMORY_TOKEN_FILE = os.environ.get("SPARK_MEMORY_TOKEN_FILE", "/etc/spark/memory/token")
VOICE_SPOOL_DIR = os.environ.get("SPARK_VOICE_SPOOL_DIR", "/var/lib/spark/voice-spool")
VOICE_CAPTURE_ENABLED = os.environ.get("SPARK_VOICE_CAPTURE_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")
VOICE_EXCLUDED_SOURCES = {"ppl-lecture", "ppl", "aecp", "healthcheck", "synthetic-test"}
# A chat model that served a request within SWAP_GRACE_SEC is not evicted for
# another one; that request gets 503 + Retry-After instead. Without this, two
# clients of different models (or one retrying client) make the router swap
# back and forth, and every swap to qwen3.8-flash-next costs ~15 min.
SWAP_GRACE_SEC = int(os.environ.get("ROUTER_SWAP_GRACE_SEC", "600"))
# Default reasoning_effort injected for reasoning models when the client
# sends none (empty = leave as-is). Keeps visible content from being eaten
# by the hidden think block on small token budgets.
DEFAULT_EFFORT = os.environ.get("ROUTER_DEFAULT_EFFORT", "low").strip().lower() or None

# API keys (Authorization: Bearer <key> or x-api-key). ROUTER_AUTH:
#   external - required for clients outside the tailnet (Tailscale Funnel)
#   all      - required for everyone except local processes on the Spark
#   off      - no authentication
# Requests through `tailscale serve`/Funnel carry X-Forwarded-For with the real
# peer (tailscaled overwrites client values; the last entry is trusted anyway).
# Direct loopback connections without it are local processes.
API_KEYS = [k.strip() for k in os.environ.get("ROUTER_API_KEYS", "").split(",") if k.strip()]
AUTH_MODE = os.environ.get("ROUTER_AUTH", "external" if API_KEYS else "off")
OPEN_PATHS = {"/health"}
# Trusted peers that may call without an API key under ROUTER_AUTH=external:
# the lean-mesh WireGuard subnet (10.88.0.0/16) and the legacy Tailscale CGNAT ranges.
_TAILNET = (ipaddress.ip_network("10.88.0.0/16"), ipaddress.ip_network("100.64.0.0/10"), ipaddress.ip_network("fd7a:115c:a1e0::/48"))
ROUTER_PORT = 8000

CHATML_STOP_SEQUENCES = (
    "<|im_end|>",
    "<|im_start|>",
    "</|im_end|>",
    "</|im_start|>",
)

MODELS = {
    "qwen3.8-flash-next": "qwen3.8-flash-next", "qwen3.8-flash": "qwen3.8-flash-next",
    "flash-next": "qwen3.8-flash-next", "qwen3.8-flash-next-uncensored": "qwen3.8-flash-next",
    "orcarouter/qwen3.8-flash-next-uncensored-nvfp4": "qwen3.8-flash-next",
    "qwen": "qwen3.8-flash-next", "qwen3.8": "qwen3.8-flash-next",

    "leanstral-1.5": "leanstral-1.5", "leanstral": "leanstral-1.5", "leanstral-15": "leanstral-1.5",
    "leanstral-1-5": "leanstral-1.5", "leanstral-1.5-119b": "leanstral-1.5",
    "leanstral-1.5-119b-a6b": "leanstral-1.5", "leanstral-1.5-nvfp4": "leanstral-1.5",
    "lean4": "leanstral-1.5", "proving": "leanstral-1.5",

    "glm-5.3-flash": "glm-5.3-flash", "glm": "glm-5.3-flash", "glm-5.3": "glm-5.3-flash",
    "glm5": "glm-5.3-flash", "glm-5.3-flash-uncensored": "glm-5.3-flash",
    "orcarouter/glm-5.3-flash-uncensored-nvfp4": "glm-5.3-flash",
}

VALID_MODELS = {"qwen3.8-flash-next", "leanstral-1.5", "glm-5.3-flash"}

MODEL_INFO = [
    {"id": "qwen3.8-flash-next", "object": "model", "canonical": "qwen3.8-flash-next"},
    {"id": "leanstral-1.5-119b-a6b", "object": "model", "canonical": "leanstral-1.5"},
    {"id": "glm-5.3-flash", "object": "model", "canonical": "glm-5.3-flash"},
    {"id": ASR_MODEL, "object": "model", "canonical": ASR_MODEL, "type": "audio-transcription"},
]

class Router:
    def __init__(self):
        self.current = None
        self.last_used = {}  # canonical model -> time of last request
        self.lock = threading.Lock()
        # Keep model selection and the corresponding inference atomic. The
        # HTTP server is threaded so health/catalog requests stay responsive,
        # but a second chat must not swap the backend during an active request.
        self.chat_lock = threading.Lock()
        self._detect()

    def _detect(self):
        try:
            r = subprocess.run(["bash", "/opt/spark/inference/swap-model.sh", "status"],
                             capture_output=True, text=True, timeout=10)
            d = json.loads(r.stdout)
            self.current = d.get("model")
            print(f"[Router] Current model: {self.current}")
        except: pass

    def resolve(self, model):
        return MODELS.get(model, MODELS.get(model.lower(), model))

    def _backend_alive(self):
        try:
            req = urllib.request.Request(f"http://localhost:{BACKEND_PORT}/health", method="GET")
            with urllib.request.urlopen(req, timeout=5) as r:
                return 200 <= r.status < 300
        except Exception:
            return False

    def ensure(self, model):
        name = self.resolve(model)
        if name not in VALID_MODELS:
            return False, f"Unknown model: {model}"

        with self.lock:
            if self.current == name:
                if self._backend_alive():
                    return True, None
                # Backend died out from under us (crash, OOM, watchdog kill).
                # Drop the stale state and fall through to relaunch it.
                print(f"[Router] Backend for {name} is unreachable; relaunching...")
                self.current = None

            if self.current is not None and SWAP_GRACE_SEC > 0:
                idle = time.time() - self.last_used.get(self.current, 0)
                if idle < SWAP_GRACE_SEC:
                    wait = int(SWAP_GRACE_SEC - idle) + 1
                    print(f"[Router] Refusing swap {self.current} -> {name}: in use {idle:.0f}s ago")
                    return False, {"busy": wait, "message":
                        f"{self.current} is loaded and in use; {name} can be loaded in {wait}s "
                        f"(only one chat model fits in memory)"}

            print(f"[Router] Swapping to {name}...")
            t0 = time.time()
            try:
                env = os.environ.copy()
                env["LLAMA_PORT"] = str(BACKEND_PORT)
                r = subprocess.run(["bash", "/opt/spark/inference/swap-model.sh", name],
                                 capture_output=True, text=True, timeout=1800, env=env)
                d = json.loads(r.stdout.strip().split('\n')[-1])
                if d.get("status") == "ready":
                    self.current = name
                    self.last_used[name] = time.time()
                    print(f"[Router] Ready in {time.time()-t0:.1f}s")
                    return True, None
                return False, d.get("message", f"Swap failed: {r.stdout}")
            except Exception as e:
                return False, str(e)

    @contextlib.contextmanager
    def in_use(self):
        """Refresh the active model's last-use time when a served request ends
        (refused requests must not extend the grace period)."""
        use = {"served": False}
        try:
            yield use
        finally:
            if use["served"] and self.current:
                self.last_used[self.current] = time.time()

    def forward(self, path, method, headers, body):
        req = urllib.request.Request(f"http://localhost:{BACKEND_PORT}{path}", data=body, method=method)
        for k, v in headers.items():
            if k.lower() not in ('host', 'content-length'):
                req.add_header(k, v)
        req.add_header('Content-Type', 'application/json')
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                response = r.read()
                if '/chat/completions' in path:
                    response = strip_chatml_sentinels(response)
                return r.status, dict(r.headers), response
        except urllib.error.HTTPError as e:
            return e.code, {}, e.read()
        except urllib.error.URLError as e:
            # Connection refused / backend down: clear stale state so the next
            # chat request relaunches it and /v1/models stops reporting it active.
            self.current = None
            return 503, {}, json.dumps({"error": f"backend unavailable: {e.reason}"}).encode()
        except Exception as e:
            return 500, {}, json.dumps({"error": str(e)}).encode()

    def open_stream(self, path, method, headers, body):
        """Open an upstream response without buffering its SSE body."""
        req = urllib.request.Request(f"http://localhost:{BACKEND_PORT}{path}", data=body, method=method)
        for k, v in headers.items():
            if k.lower() not in ('host', 'content-length', 'connection'):
                req.add_header(k, v)
        req.add_header('Content-Type', 'application/json')
        return urllib.request.urlopen(req, timeout=1800)

router = Router()


def _read_memory_api_token():
    tok = os.environ.get("MEMORY_API_TOKEN", "").strip()
    if tok:
        return tok
    try:
        return Path(MEMORY_TOKEN_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _parse_audio_multipart(content_type, body_bytes):
    """Extract audio bytes, filename, mime_type, and form fields from multipart/form-data."""
    if not content_type or "multipart/form-data" not in content_type.lower():
        return None
    try:
        header_bytes = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("latin-1", errors="ignore")
        msg = BytesParser(policy=_email_default_policy).parsebytes(header_bytes + body_bytes)
        if not msg.is_multipart():
            return None
        fields = {}
        file_info = None
        for part in msg.iter_parts():
            disp = part.get_content_disposition()
            if disp != "form-data":
                continue
            name = part.get_param("name", header="content-disposition")
            filename = part.get_filename()
            payload = part.get_payload(decode=True) or b""
            if name == "file" or filename is not None:
                if file_info is None:
                    file_info = {
                        "audio_bytes": payload,
                        "filename": filename or "audio.wav",
                        "mime_type": part.get_content_type() or "application/octet-stream",
                    }
            elif name:
                fields[name] = payload.decode("utf-8", errors="replace").strip()
        if file_info is None or not file_info["audio_bytes"]:
            return None
        file_info["fields"] = fields
        return file_info
    except Exception as exc:
        print(f"[Router] Multipart parse warning: {exc}")
        return None


def _spool_voice_request(audio_bytes, meta):
    """Persist raw audio + metadata to shared spool before ASR (Raw-First)."""
    req_id = meta["request_id"]
    ext = Path(meta.get("original_filename") or "audio.wav").suffix.lower()
    if not ext or len(ext) > 10 or not ext.startswith("."):
        ext = ".bin"
    spool = Path(VOICE_SPOOL_DIR)
    try:
        spool.mkdir(parents=True, exist_ok=True)
        audio_name = f"{req_id}{ext}"
        audio_tmp = spool / f".{audio_name}.tmp"
        audio_final = spool / audio_name
        audio_tmp.write_bytes(audio_bytes)
        os.replace(audio_tmp, audio_final)
        meta_copy = {**meta, "audio_file": audio_name}
        meta_tmp = spool / f".{req_id}.json.tmp"
        meta_final = spool / f"{req_id}.json"
        meta_tmp.write_text(json.dumps(meta_copy, ensure_ascii=False), encoding="utf-8")
        os.replace(meta_tmp, meta_final)
        return audio_final, meta_final
    except Exception as exc:
        print(f"[Router] Voice spool write warning ({req_id}): {exc}")
        return None, None


def _ship_voice_to_memory_async(audio_bytes, meta, spool_audio, spool_meta):
    def _worker():
        tok = _read_memory_api_token()
        if not tok:
            print(f"[Router] MEMORY_API_TOKEN unavailable; keeping {meta['request_id']} in spool")
            return
        boundary = f"sparkvoice{uuid.uuid4().hex}"
        parts = []
        meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="metadata_json"\r\n\r\n'.encode("utf-8")
            + meta_bytes
            + b"\r\n"
        )
        fname = (meta.get("original_filename") or "audio.wav").replace('"', "_")
        mtype = meta.get("mime_type") or "application/octet-stream"
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{fname}"\r\nContent-Type: {mtype}\r\n\r\n'.encode("utf-8")
            + audio_bytes
            + b"\r\n"
            + f"--{boundary}--\r\n".encode("utf-8")
        )
        body = b"".join(parts)
        req = urllib.request.Request(
            f"{MEMORY_GATEWAY_URL}/api/voice/ingest",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {tok}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if 200 <= resp.status < 300:
                    if spool_meta is not None:
                        Path(spool_meta).unlink(missing_ok=True)
                    if spool_audio is not None:
                        Path(spool_audio).unlink(missing_ok=True)
                    return
        except Exception as exc:
            print(f"[Router] Voice ingest handoff deferred to spool ({meta['request_id']}): {exc}")

    threading.Thread(target=_worker, daemon=True).start()


def _wait_for_asr_healthy(timeout_sec=ASR_HEALTH_WAIT_SEC):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{ASR_PORT}/health", timeout=3) as r:
                if 200 <= r.status < 300:
                    return True
        except Exception:
            pass
        time.sleep(2.0)
    return False

def strip_chatml_sentinels(body):
    for sentinel in CHATML_STOP_SEQUENCES:
        body = body.replace(sentinel.encode(), b"")
    return body

class Handler(BaseHTTPRequestHandler):
    def log_message(self, f, *a):
        src = self.headers.get('X-Forwarded-For', '-') if self.headers else '-'
        ua = (self.headers.get('User-Agent', '-') if self.headers else '-')[:60]
        model = getattr(self, '_req_model', None)
        extra = f" auth={getattr(self, '_auth', '-')}"
        dbg = getattr(self, '_req_dbg', None)
        if dbg:
            extra += " " + dbg
        extra += f" model={model}" if model is not None else ""
        why = getattr(self, '_req_error', None)
        if why:
            extra += f" error={why[:80]!r}"
        print(f"[{time.strftime('%H:%M:%S')}] {f % a} src={src} ua={ua}{extra}")

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')

    def _peer(self):
        """(ip, via_proxy): the real client address and whether it came through tailscale serve."""
        fwd = self.headers.get('X-Forwarded-For')
        if fwd:
            return fwd.split(',')[-1].strip(), True
        return self.client_address[0], False

    def _key_state(self):
        """'key' (valid), 'badkey' or 'nokey', for the access log."""
        auth = self.headers.get('Authorization', '')
        key = auth[7:].strip() if auth.lower().startswith('bearer ') else self.headers.get('x-api-key', '').strip()
        if not key:
            return 'nokey'
        return 'key' if any(hmac.compare_digest(key, k) for k in API_KEYS) else 'badkey'

    def _authorized(self):
        self._auth = self._key_state()
        if AUTH_MODE == "off" or self.path.split('?')[0] in OPEN_PATHS:
            return True
        ip, via_proxy = self._peer()
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            addr = None
        if addr is not None and addr.is_loopback and not via_proxy:
            return True  # local process on the Spark
        if AUTH_MODE == "external" and addr is not None and any(addr in n for n in _TAILNET):
            return True
        return self._auth == 'key'

    def _reject(self):
        self._req_error = "missing or invalid API key"
        body = json.dumps({"error": {"message": "Missing or invalid API key. Send Authorization: Bearer <key>.",
                                     "type": "invalid_request_error", "code": "invalid_api_key"}}).encode()
        self.send_response(401)
        self._cors()
        self.send_header('Content-Type', 'application/json')
        self.send_header('WWW-Authenticate', 'Bearer')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        if not self._authorized():
            return self._reject()
        self.send_response(200)
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if not self._authorized():
            return self._reject()
        if self.path == '/v1/models':
            data = {"object": "list", "data": [
                {**m, "active": router.current == m["canonical"]} for m in MODEL_INFO
            ]}
            self.send_response(200)
            self._cors()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        elif self.path in ('/', '/health'):
            self.send_response(200)
            self._cors()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "current_model": router.current}).encode())
        else:
            s, h, b = router.forward(self.path, 'GET', dict(self.headers), None)
            self.send_response(s)
            self._cors()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b)

    def _proxy_audio(self):
        """Forward /v1/audio/* to vllm-asr with temp-file replay on 5xx/restart,
        concurrency gating, PPL exclusion, and Raw-First personal voice capture."""
        import http.client

        t0 = time.perf_counter()
        length = int(self.headers.get('Content-Length', 0))
        content_type = self.headers.get('Content-Type', '')
        audio_source = (self.headers.get('X-Spark-Audio-Source') or '').strip().lower()
        opt_out = (self.headers.get('X-Spark-Voice-Capture') or '').strip().lower() in ('0', 'false', 'off', 'no')
        is_private_hdr = (self.headers.get('X-Spark-Voice-Private') or '').strip().lower() in ('1', 'true', 'yes', 'on')
        client_ip, _ = self._peer()
        ua = (self.headers.get('User-Agent') or '')[:240]
        self._req_model = ASR_MODEL

        req_tmp = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode='w+b')
        try:
            left = length
            while left > 0:
                chunk = self.rfile.read(min(65536, left))
                if not chunk:
                    break
                req_tmp.write(chunk)
                left -= len(chunk)
            actual_len = req_tmp.tell()

            should_capture = (
                VOICE_CAPTURE_ENABLED
                and not opt_out
                and audio_source not in VOICE_EXCLUDED_SOURCES
                and self.path.split('?')[0].rstrip('/') == '/v1/audio/transcriptions'
                and actual_len > 0
                and actual_len <= 50 * 1024 * 1024
            )

            parsed_mp = None
            req_id = f"vreq-{uuid.uuid4().hex[:16]}"
            created_iso = datetime.now(timezone.utc).isoformat()
            spool_audio = None
            spool_meta = None
            base_meta = None

            if should_capture:
                req_tmp.seek(0)
                raw_req_bytes = req_tmp.read()
                parsed_mp = _parse_audio_multipart(content_type, raw_req_bytes)
                if parsed_mp is not None:
                    fields = parsed_mp.get("fields") or {}
                    req_lang = fields.get("language")
                    base_meta = {
                        "request_id": req_id,
                        "original_filename": parsed_mp["filename"],
                        "mime_type": parsed_mp["mime_type"],
                        "requested_language": req_lang if req_lang else None,
                        "detected_language": None,
                        "language_source": "auto" if (not req_lang or req_lang.lower() == "auto") else "explicit",
                        "request_prompt": fields.get("prompt", ""),
                        "hotwords": fields.get("hotwords", ""),
                        "model": fields.get("model") or ASR_MODEL,
                        "initial_transcript": "",
                        "client_source": audio_source or "personal-voice",
                        "user_agent": ua,
                        "client_ip": client_ip,
                        "status_code": 0,
                        "latency_ms": 0.0,
                        "is_private": is_private_hdr,
                        "created_at": created_iso,
                        "metadata": {"response_format": fields.get("response_format", "json")},
                    }
                    # Raw-First: persist to disk spool BEFORE calling vllm-asr
                    spool_audio, spool_meta = _spool_voice_request(parsed_mp["audio_bytes"], base_meta)

            conn = None
            r = None
            err_body = None
            last_exc = None

            with ASR_SEMAPHORE:
                for attempt in range(ASR_MAX_RETRIES):
                    if conn is not None:
                        with contextlib.suppress(Exception):
                            conn.close()
                        conn = None
                    try:
                        req_tmp.seek(0)
                        conn = http.client.HTTPConnection('127.0.0.1', ASR_PORT, timeout=600)
                        conn.putrequest('POST', self.path, skip_accept_encoding=True)
                        for k, v in self.headers.items():
                            if k.lower() not in ('host', 'connection', 'transfer-encoding', 'content-length'):
                                conn.putheader(k, v)
                        conn.putheader('Content-Length', str(actual_len))
                        conn.endheaders()
                        while True:
                            buf = req_tmp.read(65536)
                            if not buf:
                                break
                            conn.send(buf)
                        r = conn.getresponse()
                        if r.status in (500, 502, 503, 504) and attempt + 1 < ASR_MAX_RETRIES:
                            err_body = r.read()
                            print(f"[Router] ASR returned {r.status} on attempt {attempt + 1}/{ASR_MAX_RETRIES}; waiting for healthy :8002...")
                            conn.close()
                            conn = None
                            r = None
                            time.sleep(1.5)
                            _wait_for_asr_healthy()
                            continue
                        break
                    except OSError as e:
                        last_exc = e
                        if conn is not None:
                            with contextlib.suppress(Exception):
                                conn.close()
                            conn = None
                        if attempt + 1 < ASR_MAX_RETRIES:
                            print(f"[Router] ASR transport error ({e}) on attempt {attempt + 1}/{ASR_MAX_RETRIES}; waiting for healthy :8002...")
                            time.sleep(1.5)
                            _wait_for_asr_healthy()
                            continue

            if r is None:
                self._req_error = f"asr unavailable: {last_exc or err_body}"
                self.send_response(503)
                self._cors()
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                msg = f"transcription backend unavailable: {last_exc}" if last_exc else "transcription backend failed after retries"
                self.wfile.write(json.dumps({"error": {"message": msg}}).encode())
                return

            try:
                resp_ctype = r.getheader('Content-Type', 'application/json')
                detected_lang = r.getheader('X-Spark-Detected-Language')
                lang_source = r.getheader('X-Spark-Language-Source')
                dur_hdr = r.getheader('X-Spark-Audio-Duration-Seconds')
                self._req_dbg = f"src={audio_source or 'voice'} lang={detected_lang or '-'}"

                self.send_response(r.status)
                self._cors()
                self.send_header('Content-Type', resp_ctype)
                if detected_lang:
                    self.send_header('X-Spark-Detected-Language', detected_lang)
                if lang_source:
                    self.send_header('X-Spark-Language-Source', lang_source)

                if 'text/event-stream' in (resp_ctype or ''):
                    self.send_header('Cache-Control', 'no-cache')
                    self.end_headers()
                    sse_chunks = []
                    while chunk := r.read1(8192):
                        if base_meta is not None and len(sse_chunks) < 256:
                            sse_chunks.append(chunk)
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    if base_meta is not None and parsed_mp is not None:
                        transcript_parts = []
                        for raw_line in b"".join(sse_chunks).decode("utf-8", errors="ignore").splitlines():
                            if raw_line.startswith("data: ") and raw_line[6:].strip() != "[DONE]":
                                with contextlib.suppress(Exception):
                                    ev = json.loads(raw_line[6:])
                                    for ch in ev.get("choices") or []:
                                        delta = (ch.get("delta") or {}).get("content")
                                        if delta:
                                            transcript_parts.append(delta)
                        base_meta["initial_transcript"] = "".join(transcript_parts).strip()
                        base_meta["status_code"] = r.status
                        base_meta["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
                        if detected_lang:
                            base_meta["detected_language"] = detected_lang
                        if lang_source:
                            base_meta["language_source"] = lang_source
                        if dur_hdr:
                            with contextlib.suppress(ValueError):
                                base_meta["duration_seconds"] = float(dur_hdr)
                        spool_audio, spool_meta = _spool_voice_request(parsed_mp["audio_bytes"], base_meta)
                        _ship_voice_to_memory_async(parsed_mp["audio_bytes"], base_meta, spool_audio, spool_meta)
                    return

                body = r.read()
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

                if base_meta is not None and parsed_mp is not None:
                    transcript_txt = ""
                    if 'application/json' in (resp_ctype or '').lower():
                        with contextlib.suppress(Exception):
                            payload = json.loads(body.decode("utf-8", errors="replace"))
                            transcript_txt = str(payload.get("text") or "").strip()
                            if not detected_lang and payload.get("language"):
                                detected_lang = str(payload.get("language"))
                            if not dur_hdr and isinstance(payload.get("usage"), dict):
                                sec = payload["usage"].get("seconds")
                                if sec is not None:
                                    dur_hdr = str(sec)
                    else:
                        transcript_txt = body.decode("utf-8", errors="replace").strip()

                    base_meta["initial_transcript"] = transcript_txt
                    base_meta["status_code"] = r.status
                    base_meta["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
                    base_meta["detected_language"] = detected_lang or base_meta.get("requested_language")
                    if lang_source:
                        base_meta["language_source"] = lang_source
                    if dur_hdr:
                        with contextlib.suppress(ValueError):
                            base_meta["duration_seconds"] = float(dur_hdr)
                    spool_audio, spool_meta = _spool_voice_request(parsed_mp["audio_bytes"], base_meta)
                    _ship_voice_to_memory_async(parsed_mp["audio_bytes"], base_meta, spool_audio, spool_meta)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                if conn is not None:
                    conn.close()
        finally:
            req_tmp.close()

    def do_POST(self):
        if not self._authorized():
            return self._reject()
        try:
            with Foreground():
                if self.path.startswith('/v1/audio/'):
                    self._proxy_audio()
                else:
                    self._post_with_gpu()
        except OSError:
            self.send_error(503, "GPU admission unavailable")

    def _post_with_gpu(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))

        if '/chat/completions' in self.path:
            try:
                data = json.loads(body)
                requested = data.get('model', 'qwen3.8-flash-next')
                self._req_model = requested
                try:
                    _msgs = data.get('messages') or []
                    self._req_dbg = ("mt=%s stream=%s effort=%s n_msgs=%d last_len=%d" % (
                        data.get('max_tokens'), bool(data.get('stream')),
                        data.get('reasoning_effort') or (data.get('chat_template_kwargs') or {}).get('reasoning_effort'),
                        len(_msgs), len(str((_msgs[-1].get('content') if _msgs else '') or ''))))
                except Exception:
                    pass
            except json.JSONDecodeError as e:
                self._req_error = f"invalid JSON: {e}"
                self.send_response(400)
                self._cors()
                self.end_headers()
                self.wfile.write(json.dumps({"error": {"message": str(e)}}).encode())
                return
            with router.chat_lock, router.in_use() as use:
                ok, err = router.ensure(requested)
                if not ok:
                    busy = isinstance(err, dict)
                    msg = err["message"] if busy else err
                    self._req_error = msg  # set before send_response, which writes the log line
                    self.send_response(503 if busy else 400)
                    self._cors()
                    self.send_header('Content-Type', 'application/json')
                    if busy:
                        self.send_header('Retry-After', str(err["busy"]))
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": {"message": msg}}).encode())
                    return
                router.last_used[router.current] = time.time()
                use["served"] = True
                # Rewrite the alias to the backend's served model name. vLLM
                # validates the model field against --served-model-name and 404s
                # on anything else, so aliases like "qwen"/"flash-next" must become
                # the canonical id before forwarding. (llama.cpp ignores it.)
                data['model'] = router.resolve(requested)
                # qwen3.8-flash-next and glm-5.3-flash are reasoning models: they
                # open a hidden <think> block. With no effort set they think at the
                # top budget, so a request with a small max_tokens spends it all on
                # reasoning and returns EMPTY content (finish_reason=length). Default
                # to DEFAULT_EFFORT (low) when the client sends none, matching Orb's
                # own Ask client. An explicit reasoning_effort still wins.
                effort = data.get('reasoning_effort')
                if effort is None and DEFAULT_EFFORT and \
                        not (data.get('chat_template_kwargs') or {}).get('reasoning_effort'):
                    effort = DEFAULT_EFFORT
                if data['model'] == 'glm-5.3-flash':
                    # SparkLab reads the GLM budget (low/high/max) only from
                    # chat_template_kwargs; without it the template thinks at max.
                    if effort is not None:
                        kwargs = data.setdefault('chat_template_kwargs', {})
                        kwargs.setdefault('reasoning_effort', {
                            'none': 'low', 'minimal': 'low', 'low': 'low', 'medium': 'high',
                            'high': 'high', 'xhigh': 'max', 'max': 'max'}.get(str(effort).lower(), 'max'))
                        data.pop('reasoning_effort', None)
                elif data['model'] == 'qwen3.8-flash-next' and effort is not None:
                    # The qwen3.8 chat template only accepts low/medium/xhigh.
                    data['reasoning_effort'] = {'none': 'low', 'minimal': 'low',
                        'high': 'xhigh'}.get(str(effort).lower(), str(effort).lower())
                body = json.dumps(data).encode()
                if data.get('stream'):
                    try:
                        with router.open_stream(self.path, 'POST', dict(self.headers), body) as upstream:
                            self.send_response(upstream.status)
                            self._cors()
                            self.send_header('Content-Type', upstream.headers.get('Content-Type', 'text/event-stream'))
                            self.send_header('Cache-Control', 'no-cache')
                            self.end_headers()
                            read_chunk = getattr(upstream, 'read1', upstream.read)
                            while chunk := read_chunk(8192):
                                self.wfile.write(chunk)
                                self.wfile.flush()
                        return
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    except urllib.error.HTTPError as e:
                        self.send_response(e.code)
                        self._cors()
                        self.end_headers()
                        self.wfile.write(e.read())
                        return
                s, h, b = router.forward(self.path, 'POST', dict(self.headers), body)
        else:
            s, h, b = router.forward(self.path, 'POST', dict(self.headers), body)
        self.send_response(s)
        self._cors()
        self.send_header('Content-Type', h.get('Content-Type', 'application/json'))
        self.end_headers()
        self.wfile.write(b)

if __name__ == "__main__":
    print("=" * 50)
    print("Multi-Model Router for DGX Spark")
    print("=" * 50)
    print(f"Models: qwen3.8-flash-next, leanstral-1.5-119b-a6b, glm-5.3-flash (chat), {ASR_MODEL} (/v1/audio/transcriptions)")
    print(f"Aliases: qwen, qwen3.8, qwen3.8-flash, flash-next, leanstral, lean4, proving, glm")
    print(f"Current: {router.current}")
    print(f"Listening: http://0.0.0.0:{ROUTER_PORT}")
    print(f"Mesh: http://10.88.0.2:{ROUTER_PORT}/v1/chat/completions")
    print("=" * 50)
    router_heartbeat()
    ThreadingHTTPServer(('0.0.0.0', ROUTER_PORT), Handler).serve_forever()
