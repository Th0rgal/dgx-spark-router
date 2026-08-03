import json
import threading
import time
import unittest
import urllib.request
from unittest import mock

import router


class RouterHelpersTest(unittest.TestCase):
    def test_add_chatml_stops_preserves_client_stops(self):
        data = {"stop": "DONE"}

        router.add_chatml_stops(data)

        self.assertEqual(data["stop"][0], "DONE")
        self.assertEqual(set(data["stop"][1:]), set(router.CHATML_STOP_SEQUENCES))

    def test_strip_chatml_sentinels_handles_json_and_sse(self):
        body = (
            b'data: {"choices":[{"delta":{"content":"OK<|im_end|>"}}]}\n\n'
            b'data: [DONE]\n\n'
        )

        cleaned = router.strip_chatml_sentinels(body)

        self.assertNotIn(b"im_end", cleaned)
        self.assertIn(b'"content":"OK"', cleaned)
        self.assertIn(b"data: [DONE]", cleaned)


class ThreadedCatalogTest(unittest.TestCase):
    def test_models_remains_available_during_chat(self):
        chat_started = threading.Event()
        release_chat = threading.Event()

        def slow_forward(path, method, headers, body):
            if "/chat/completions" in path:
                chat_started.set()
                release_chat.wait(timeout=2)
                response = {
                    "choices": [{"message": {"role": "assistant", "content": "OK<|im_end|>"}}]
                }
                return 200, {"Content-Type": "application/json"}, router.strip_chatml_sentinels(
                    json.dumps(response).encode()
                )
            raise AssertionError(path)

        fake_router = mock.Mock()
        fake_router.current = "leanstral-1.5"
        fake_router.chat_lock = threading.Lock()
        fake_router.ensure.return_value = (True, None)
        fake_router.resolve.return_value = "leanstral-1.5"
        fake_router.forward.side_effect = slow_forward

        server = router.ThreadingHTTPServer(("127.0.0.1", 0), router.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}"

        def send_chat():
            request = urllib.request.Request(
                f"{base_url}/v1/chat/completions",
                data=json.dumps(
                    {"model": "leanstral-1.5-119b-a6b", "messages": [{"role": "user", "content": "hi"}]}
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=3) as response:
                return response.read()

        with mock.patch.object(router, "router", fake_router):
            chat_thread = threading.Thread(target=send_chat)
            chat_thread.start()
            self.assertTrue(chat_started.wait(timeout=1))

            started = time.monotonic()
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=1) as response:
                catalog = json.load(response)
            elapsed = time.monotonic() - started

            self.assertLess(elapsed, 0.5)
            self.assertTrue(any(model["id"] == "leanstral-1.5-119b-a6b" for model in catalog["data"]))
            release_chat.set()
            chat_thread.join(timeout=2)

        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
