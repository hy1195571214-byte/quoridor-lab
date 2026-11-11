"""Serve the production website locally, with optional /api proxy to the model service."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler
import argparse
import json

ROOT = Path(__file__).resolve().parents[1]
DIRECT = build_opener(ProxyHandler({}))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "frontend/dist/client"), **kwargs)

    def proxy(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 256_000:
            self.send_error(413)
            return
        body = self.rfile.read(length) if length else None
        request = Request("http://127.0.0.1:8765" + self.path[4:], data=body,
                          headers={"Content-Type": self.headers.get("Content-Type", "application/json")},
                          method=self.command)
        try:
            response = DIRECT.open(request, timeout=60)
        except HTTPError as error:
            response = error
        except URLError:
            self.send_response(502)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Start the local model service on port 8765"}).encode())
            return
        with response:
            self.send_response(response.status)
            self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
            self.end_headers()
            self.wfile.write(response.read())

    def do_GET(self):
        if self.path.startswith("/api/"):
            return self.proxy()
        requested = Path(self.translate_path(self.path))
        if not requested.exists() and "." not in Path(self.path.split("?")[0]).name:
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/"):
            return self.proxy()
        self.send_error(405)

    do_PUT = do_DELETE = do_POST


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not (ROOT / "frontend/dist/client/index.html").exists():
        parser.error("Run npm ci && npm run build in frontend first, or open frontend/offline/index.html")
    print(f"Local website: http://127.0.0.1:{args.port}/local", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
