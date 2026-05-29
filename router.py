#!/usr/bin/env python3
"""Multi-Model OpenAI-Compatible Router for DGX Spark"""
import os, json, subprocess, time, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request, urllib.error

BACKEND_PORT = 8001
ROUTER_PORT = 8000

MODELS = {
    "minimax": "minimax", "minimax-m2.1": "minimax", "tool-calling": "minimax", "coding": "minimax",
    "gpt-oss": "gpt-oss", "gpt-oss-120b": "gpt-oss", "scientific": "gpt-oss", "writing": "gpt-oss",
    "glm-flash": "glm-flash", "glm-4.7-flash": "glm-flash", "fast": "glm-flash",
    "deepseek": "deepseek", "deepseek-v4": "deepseek", "deepseek-v4-flash": "deepseek",
    "deepseek-v4-flash-spark": "deepseek", "reasoning": "deepseek", "thinking": "deepseek",
}

VALID_MODELS = {"minimax", "gpt-oss", "glm-flash", "deepseek"}

MODEL_INFO = [
    {"id": "minimax-m2.1", "object": "model", "canonical": "minimax"},
    {"id": "gpt-oss-120b", "object": "model", "canonical": "gpt-oss"},
    {"id": "glm-4.7-flash", "object": "model", "canonical": "glm-flash"},
    {"id": "deepseek-v4-flash", "object": "model", "canonical": "deepseek"},
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

    def ensure(self, model):
        name = MODELS.get(model, MODELS.get(model.lower(), model))
        if name not in VALID_MODELS:
            return False, f"Unknown model: {model}"

        with self.lock:
            if self.current == name:
                return True, None

            print(f"[Router] Swapping to {name}...")
            t0 = time.time()
            try:
                env = os.environ.copy()
                env["LLAMA_PORT"] = str(BACKEND_PORT)
                r = subprocess.run(["bash", os.path.expanduser("~/swap-model.sh"), name],
                                 capture_output=True, text=True, timeout=600, env=env)
                d = json.loads(r.stdout)
                if d.get("status") == "ready":
                    self.current = name
                    print(f"[Router] Ready in {time.time()-t0:.1f}s")
                    return True, None
                return False, f"Swap failed: {r.stdout}"
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
                ok, err = router.ensure(data.get('model', 'minimax'))
                if not ok:
                    self.send_response(400)
                    self._cors()
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": {"message": err}}).encode())
                    return
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
    print(f"Models: minimax-m2.1, gpt-oss-120b, glm-4.7-flash, deepseek-v4-flash")
    print(f"Aliases: tool-calling, coding, scientific, writing, fast, reasoning, thinking")
    print(f"Current: {router.current}")
    print(f"Listening: http://0.0.0.0:{ROUTER_PORT}")
    print(f"Public: https://spark-de79.gazella-vector.ts.net/v1/chat/completions")
    print("=" * 50)
    HTTPServer(('0.0.0.0', ROUTER_PORT), Handler).serve_forever()
