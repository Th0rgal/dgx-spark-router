#!/usr/bin/env python3
"""Multi-Model OpenAI-Compatible Router for DGX Spark"""
import os, json, subprocess, time, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request, urllib.error

BACKEND_PORT = 8001
ROUTER_PORT = 8000

MODELS = {
    "gpt-oss": "gpt-oss", "gpt-oss-120b": "gpt-oss", "scientific": "gpt-oss", "writing": "gpt-oss",
    "leanstral": "leanstral", "leanstral-2603": "leanstral", "lean4": "leanstral", "proving": "leanstral",
    "nemotron-3-super": "nemotron-3-super", "nemotron": "nemotron-3-super", "nemotron-3": "nemotron-3-super",
    "nemotron-super": "nemotron-3-super", "nemotron-3-super-120b": "nemotron-3-super",
    "reasoning": "nemotron-3-super", "thinking": "nemotron-3-super",
    "qwen3.6": "qwen3.6", "qwen3.6-35b": "qwen3.6", "qwen3.6-35b-a3b": "qwen3.6", "qwen3-6": "qwen3.6",
    "gemma-4": "gemma-4", "gemma4": "gemma-4", "gemma": "gemma-4",
    "gemma-4-26b": "gemma-4", "gemma-4-26b-a4b": "gemma-4",
}

VALID_MODELS = {"gpt-oss", "leanstral", "nemotron-3-super", "qwen3.6", "gemma-4"}

MODEL_INFO = [
    {"id": "gpt-oss-120b", "object": "model", "canonical": "gpt-oss"},
    {"id": "leanstral-2603", "object": "model", "canonical": "leanstral"},
    {"id": "nemotron-3-super", "object": "model", "canonical": "nemotron-3-super"},
    {"id": "qwen3.6", "object": "model", "canonical": "qwen3.6"},
    {"id": "gemma-4", "object": "model", "canonical": "gemma-4"},
]

class Router:
    def __init__(self):
        self.current = None
        self.lock = threading.Lock()
        self._detect()

    def _detect(self):
        try:
            r = subprocess.run(["bash", os.path.expanduser("~/swap-model.sh"), "status"],
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
                r = subprocess.run(["bash", os.path.expanduser("~/swap-model.sh"), name],
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
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, {}, e.read()
        except urllib.error.URLError as e:
            # Connection refused / backend down: clear stale state so the next
            # chat request relaunches it and /v1/models stops reporting it active.
            self.current = None
            return 503, {}, json.dumps({"error": f"backend unavailable: {e.reason}"}).encode()
        except Exception as e:
            return 500, {}, json.dumps({"error": str(e)}).encode()

router = Router()

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

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))

        if '/chat/completions' in self.path:
            try:
                data = json.loads(body)
                requested = data.get('model', 'gpt-oss')
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
                body = json.dumps(data).encode()
            except json.JSONDecodeError as e:
                self.send_response(400)
                self._cors()
                self.end_headers()
                self.wfile.write(json.dumps({"error": {"message": str(e)}}).encode())
                return

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
    print(f"Models: gpt-oss-120b, leanstral-2603, nemotron-3-super, qwen3.6, gemma-4")
    print(f"Aliases: scientific, writing, lean4, proving, reasoning, thinking, gemma")
    print(f"Current: {router.current}")
    print(f"Listening: http://0.0.0.0:{ROUTER_PORT}")
    print(f"Public: https://spark-de79.gazella-vector.ts.net/v1/chat/completions")
    print("=" * 50)
    HTTPServer(('0.0.0.0', ROUTER_PORT), Handler).serve_forever()
