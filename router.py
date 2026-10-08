#!/usr/bin/env python3
"""Multi-Model OpenAI-Compatible Router for DGX Spark"""
import os, json, subprocess, time, threading
from gpu_idle import Foreground, router_heartbeat
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import urllib.request, urllib.error

BACKEND_PORT = 8001
# Speech-to-text backend (Cohere Transcribe NVFP4, container vllm-asr). It runs
# beside the chat model, so audio requests never trigger a model swap. They take
# the shared GPU foreground lock like chat requests (pauses vanity, not qwen).
ASR_PORT = 8002
ASR_MODEL = "cohere-transcribe"
ROUTER_PORT = 8000

CHATML_STOP_SEQUENCES = (
    "<|im_end|>",
    "<|im_start|>",
    "</|im_end|>",
    "</|im_start|>",
)

MODELS = {
    "gpt-oss": "gpt-oss", "gpt-oss-120b": "gpt-oss", "scientific": "gpt-oss", "writing": "gpt-oss",
    "leanstral": "leanstral-1.5", "leanstral-1.5": "leanstral-1.5", "leanstral-15": "leanstral-1.5",
    "leanstral-1-5": "leanstral-1.5", "leanstral-1.5-119b": "leanstral-1.5",
    "leanstral-1.5-119b-a6b": "leanstral-1.5", "leanstral-119b": "leanstral-1.5",
    "leanstral-2603": "leanstral", "lean4": "leanstral-1.5", "proving": "leanstral-1.5",
    "nemotron-3-super": "nemotron-3-super", "nemotron": "nemotron-3-super", "nemotron-3": "nemotron-3-super",
    "nemotron-super": "nemotron-3-super", "nemotron-3-super-120b": "nemotron-3-super",
    "reasoning": "nemotron-3-super", "thinking": "nemotron-3-super",

    "gemma-4": "gemma-4", "gemma4": "gemma-4", "gemma": "gemma-4",
    "gemma-4-26b": "gemma-4", "gemma-4-26b-a4b": "gemma-4",

    "qwen3.8-orca-nvfp4": "qwen3.8-orca-nvfp4", "qwen3.8": "qwen3.8-orca-nvfp4",
    "qwen3.8-27b": "qwen3.8-orca-nvfp4", "qwen3.8-27b-uncensored": "qwen3.8-orca-nvfp4",
    "orcarouter/qwen3.8-27b-uncensored-nvfp4": "qwen3.8-orca-nvfp4",
    "qwen3.8-orca-q4": "qwen3.8-orca-nvfp4", "qwen3.8-orca": "qwen3.8-orca-nvfp4",
    "qwen": "qwen3.8-orca-nvfp4",

    "qwen3.8-flash-next": "qwen3.8-flash-next", "qwen3.8-flash": "qwen3.8-flash-next",
    "flash-next": "qwen3.8-flash-next", "qwen3.8-flash-next-uncensored": "qwen3.8-flash-next",
    "orcarouter/qwen3.8-flash-next-uncensored-nvfp4": "qwen3.8-flash-next",

}

VALID_MODELS = {"gpt-oss", "leanstral", "leanstral-1.5", "nemotron-3-super", "qwen3.8-orca-nvfp4", "qwen3.8-flash-next", "gemma-4"}

MODEL_INFO = [
    {"id": "gpt-oss-120b", "object": "model", "canonical": "gpt-oss"},
    {"id": "leanstral-1.5-119b-a6b", "object": "model", "canonical": "leanstral-1.5"},
    {"id": "leanstral-2603", "object": "model", "canonical": "leanstral"},
    {"id": "nemotron-3-super", "object": "model", "canonical": "nemotron-3-super"},

    {"id": "qwen3.8-orca-nvfp4", "object": "model", "canonical": "qwen3.8-orca-nvfp4"},
    {"id": "qwen3.8-flash-next", "object": "model", "canonical": "qwen3.8-flash-next"},
    {"id": "gemma-4", "object": "model", "canonical": "gemma-4"},
    {"id": ASR_MODEL, "object": "model", "canonical": ASR_MODEL, "type": "audio-transcription"},

]

class Router:
    def __init__(self):
        self.current = None
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
                    print(f"[Router] Ready in {time.time()-t0:.1f}s")
                    return True, None
                return False, d.get("message", f"Swap failed: {r.stdout}")
            except Exception as e:
                return False, str(e)

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

def add_chatml_stops(data):
    existing = data.get("stop", [])
    if isinstance(existing, str):
        existing = [existing]
    elif not isinstance(existing, list):
        existing = []
    data["stop"] = existing + [stop for stop in CHATML_STOP_SEQUENCES if stop not in existing]

def strip_chatml_sentinels(body):
    for sentinel in CHATML_STOP_SEQUENCES:
        body = body.replace(sentinel.encode(), b"")
    return body

class Handler(BaseHTTPRequestHandler):
    def log_message(self, f, *a): print(f"[{time.strftime('%H:%M:%S')}] {f % a}")

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
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
                requested = data.get('model', 'gpt-oss')
            except json.JSONDecodeError as e:
                self.send_response(400)
                self._cors()
                self.end_headers()
                self.wfile.write(json.dumps({"error": {"message": str(e)}}).encode())
                return
            with router.chat_lock:
                ok, err = router.ensure(requested)
                if not ok:
                    self.send_response(400)
                    self._cors()
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": {"message": err}}).encode())
                    return
                # Rewrite the alias to the backend's served model name. vLLM
                # validates the model field against --served-model-name and 404s
                # on anything else, so "nemotron"/"reasoning"/etc. must become
                # the canonical id before forwarding. (llama.cpp ignores it.)
                data['model'] = router.resolve(requested)
                # Normalize OpenAI-style reasoning_effort values the qwen3.8
                # chat template rejects (it only accepts low/medium/xhigh).
                effort = data.get('reasoning_effort')
                if effort is not None:
                    mapped = {'none': 'low', 'minimal': 'low', 'high': 'xhigh'}.get(str(effort).lower())
                    if mapped:
                        data['reasoning_effort'] = mapped
                # Only the legacy Leanstral-2603 backend uses ChatML sentinels. Leanstral 1.5 uses
                # Mistral's official [TOOL_CALLS]/[ARGS] template and llama.cpp response parser.
                if data['model'] == 'leanstral':
                    add_chatml_stops(data)
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
    print(f"Models: gpt-oss-120b, leanstral-1.5-119b-a6b, leanstral-2603, nemotron-3-super, qwen3.8-orca-nvfp4, qwen3.8-flash-next, gemma-4")
    print(f"Aliases: scientific, writing, leanstral, lean4, proving, reasoning, thinking, qwen3.8, qwen3.8-orca-q4, qwen3.8-27b, qwen3.8-27b-uncensored, qwen3.8-flash, flash-next, gemma")
    print(f"Current: {router.current}")
    print(f"Listening: http://0.0.0.0:{ROUTER_PORT}")
    print(f"Public: https://spark-de79.gazella-vector.ts.net/v1/chat/completions")
    print("=" * 50)
    router_heartbeat()
    ThreadingHTTPServer(('0.0.0.0', ROUTER_PORT), Handler).serve_forever()
