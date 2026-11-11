"""Local agent adapter example. These are placeholders, not trained models.

Run from the repository root:
  PYTHONPATH=training python training/examples/serve_custom_agent.py
The example never calls the remote expert. Replace each policy's choose method
with local model inference or search and return Decision with a legal Action.
"""
import argparse
import uvicorn

from barricade_lab.api import create_app
from barricade_lab.opponents import Decision, LocalPolicy, RemoteExpert


class PlaceholderPolicy:
    def __init__(self, name):
        self.name = name
        self.baseline = LocalPolicy("shortest")

    def choose(self, state):
        # state.observation() contains board and the 209-action legality mask.
        # A real model must select a legal Action and load its own weights.
        action = self.baseline.choose(state).action
        return Decision(action, f"placeholder:{self.name}:shortest")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    agents = {name: PlaceholderPolicy(name) for name in ("dl", "rl", "search")}
    # This configures the optional expert adapter but does not contact it.
    # Only the explicitly requested /expert or remote-game paths do so.
    app = create_app(RemoteExpert(data_dir=args.data_dir), agents=agents)
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
