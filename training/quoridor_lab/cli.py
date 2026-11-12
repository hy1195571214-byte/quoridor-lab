from __future__ import annotations

import argparse
import json

from .collect import collect
from .opponents import RemoteExpert, LocalPolicy, RemoteError
from .rules import State


def main():
    parser = argparse.ArgumentParser(description="Quoridor Lab expert adapter and dataset collection")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "collect", "serve"):
        p = sub.add_parser(name)
        p.add_argument("--data-dir", default="data")
        p.add_argument("--difficulty", default="expert", choices=sorted(RemoteExpert.difficulties))
        p.add_argument("--interval", type=float, default=1.0, help="Minimum seconds between expert calls")
        p.add_argument("--timeout", type=float, default=30)
        p.add_argument("--proxy-env", action="store_true", help="Use HTTP(S)_PROXY instead of direct requests")
        p.add_argument("--no-cache", action="store_true")
    sub.choices["probe"].add_argument("--moves", default="e2", help="Comma-separated legal move history; empty string for opening")
    p = sub.choices["collect"]
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--output", default="data/collection")
    p.add_argument("--policy", choices=["shortest", "random"], default="shortest")
    p.add_argument("--opponent", choices=["remote", "shortest", "random"], default="remote")
    p.add_argument("--expert-side", choices=["1", "2", "alternate"], default="alternate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-plies", type=int, default=256)
    p.add_argument("--resume", action="store_true")
    p = sub.choices["serve"]
    p.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    remote = RemoteExpert(difficulty=args.difficulty, data_dir=args.data_dir,
                          interval=args.interval, timeout=args.timeout,
                          proxy_env=args.proxy_env, cache=not args.no_cache)
    try:
        if args.command == "probe":
            state = State.from_moves(args.moves.split(",") if args.moves else [])
            print(json.dumps(remote.choose(state).to_dict(), ensure_ascii=False, indent=2))
        elif args.command == "collect":
            opponent = remote if args.opponent == "remote" else LocalPolicy(args.opponent, args.seed)
            result = collect(opponent, output=args.output, games=args.games, policy=args.policy,
                             seed=args.seed, expert_side=args.expert_side,
                             max_plies=args.max_plies, resume=args.resume)
            summary = {k: v for k, v in result.items() if k != "episodes"}
            print(json.dumps(summary, ensure_ascii=False))
            if result["error"]:
                raise SystemExit(2)
        else:
            import uvicorn
            from .api import create_app
            uvicorn.run(create_app(remote), host="127.0.0.1", port=args.port)
    except RemoteError as error:
        print(json.dumps({"error": error.kind, "message": str(error),
                          "retryable": error.retryable}, ensure_ascii=False))
        raise SystemExit(2)
    finally:
        remote.close()


if __name__ == "__main__":
    main()
