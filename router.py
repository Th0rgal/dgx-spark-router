#!/usr/bin/env python3
"""Multi-Model OpenAI-Compatible Router for DGX Spark"""
import contextlib, hmac, ipaddress, os, json, subprocess, time, threading
from gpu_idle import Foreground, router_heartbeat
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import urllib.request, urllib.error

BACKEND_PORT = 8001
# Speech-to-text backend (Cohere Transcribe NVFP4, container vllm-asr). It runs
# beside the chat model, so audio requests never trigger a model swap. They take
# the shared GPU foreground lock like chat requests (pauses vanity, not qwen).
ASR_PORT = 8002
ASR_MODEL = "cohere-transcribe"
# A chat model that served a request within SWAP_GRACE_SEC is not evicted for
# another one; that request gets 503 + Retry-After instead. Without this, two
# clients of different models (or one retrying client) make the router swap
# back and forth, and every swap to qwen3.8-flash-next costs ~15 min.
SWAP_GRACE_SEC = int(os.environ.get("ROUTER_SWAP_GRACE_SEC", "600"))

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
_TAILNET = (ipaddress.ip_network("100.64.0.0/10"), ipaddress.ip_network("fd7a:115c:a1e0::/48"))
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
        """Stream /v1/audio/* to the ASR backend: request body and (SSE)
        response are forwarded in chunks without buffering."""
        import http.client
        length = int(self.headers.get('Content-Length', 0))
        try:
            conn = http.client.HTTPConnection('127.0.0.1', ASR_PORT, timeout=600)
            conn.putrequest('POST', self.path, skip_accept_encoding=True)
            for k, v in self.headers.items():
                if k.lower() not in ('host', 'connection', 'transfer-encoding'):
                    conn.putheader(k, v)
            conn.endheaders()
            left = length
            while left > 0:
                chunk = self.rfile.read(min(65536, left))
                if not chunk:
                    break
                conn.send(chunk)
                left -= len(chunk)
            r = conn.getresponse()
        except OSError as e:
            self.send_response(503)
            self._cors()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": f"transcription backend unavailable: {e}"}}).encode())
            return
        try:
            self.send_response(r.status)
            self._cors()
            self.send_header('Content-Type', r.getheader('Content-Type', 'application/json'))
            if 'text/event-stream' in (r.getheader('Content-Type') or ''):
                self.send_header('Cache-Control', 'no-cache')
            else:
                body = r.read()
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.end_headers()
            while chunk := r.read1(8192):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()

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
                effort = data.get('reasoning_effort')
                if data['model'] == 'glm-5.3-flash':
                    # SparkLab reads the GLM budget (low/high/max) only from
                    # chat_template_kwargs; without it the template thinks at max.
                    if effort is not None:
                        kwargs = data.setdefault('chat_template_kwargs', {})
                        kwargs.setdefault('reasoning_effort', {
                            'none': 'low', 'minimal': 'low', 'low': 'low', 'medium': 'high',
                            'high': 'high', 'xhigh': 'max', 'max': 'max'}.get(str(effort).lower(), 'max'))
                        del data['reasoning_effort']
                elif effort is not None:
                    # The qwen3.8 chat template only accepts low/medium/xhigh.
                    mapped = {'none': 'low', 'minimal': 'low', 'high': 'xhigh'}.get(str(effort).lower())
                    if mapped:
                        data['reasoning_effort'] = mapped
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
    print(f"Public: https://spark-de79.gazella-vector.ts.net/v1/chat/completions")
    print("=" * 50)
    router_heartbeat()
    ThreadingHTTPServer(('0.0.0.0', ROUTER_PORT), Handler).serve_forever()
