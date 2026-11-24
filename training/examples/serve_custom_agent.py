"""Local agent adapter: dl/rl placeholders + SearchAgent on the search slot.

Run from the repository root:
  PYTHONPATH=training .venv/bin/python training/examples/serve_custom_agent.py --port 8765 --data-dir ./handoff-local-data
"""
import argparse
import uvicorn

from quoridor_lab.api import create_app
from quoridor_lab.opponents import Decision, LocalPolicy, RemoteExpert
from quoridor_lab.search import SearchAgent


class PlaceholderPolicy:
    def __init__(self, name):
        self.name = name
        self.baseline = LocalPolicy("shortest")

    def choose(self, state):
        action = self.baseline.choose(state).action
        return Decision(action, f"placeholder:{self.name}:shortest")

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    agents = {
        "dl": PlaceholderPolicy("dl"),
        "rl": PlaceholderPolicy("rl"),
        "search": SearchAgent("medium"),
        "search_easy": SearchAgent("easy"),
        "search_medium": SearchAgent("medium"),
        "search_hard": SearchAgent("hard"),
    }
    app = create_app(RemoteExpert(data_dir=args.data_dir), agents=agents)
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()