import argparse
import runpy
import sys
import os

def _run_module(mod_path: str, args: list[str]) -> int:
    """
    Run the module as __main__ passing the arguments.
    """
    sys.argv = [mod_path] + args
    try:
        runpy.run_module(mod_path, run_name="__main__")
        return 0
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1

def main(argv=None):
    p = argparse.ArgumentParser(prog="graphx", description="Graph Exploration CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Each command captures additional arguments after '--'
    sub.add_parser("env-sunt", help="Run the entrypoint of the SUNT environment (if available)") \
        .add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("train", help="Train RLlib") \
        .add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("stats", help="Calculate statistics/averages") \
        .add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("look-amount", help="Tool lookAmount") \
        .add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("pkl-medias", help="Tool pklFilesMedias") \
        .add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("see-routes", help="Tool seeRoutesFiles") \
        .add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("view-pkl", help="Tool viewPklFiles") \
        .add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("view-graph", help="Tool viewGraph") \
        .add_argument("args", nargs=argparse.REMAINDER)

    args = p.parse_args(argv)

    # Command mapping -> module
    modmap = {
        "env-sunt": "src.envs.sunt_env",
        "train": "src.pipelines.train_rllib",
        "stats": "src.pipelines.show_dataset_stats",
        "look-amount": "src.tools.look_amount",
        "pkl-medias": "src.tools.pkl_medias",
        "see-routes": "src.tools.see_routes",
        "view-pkl": "src.tools.view_pkl",
        "view-graph": "src.viz.view_graph",
    }

    mod = modmap[args.cmd]
    passthrough = getattr(args, "args", []) or []

    # Roda o módulo com os argumentos pass-through
    return _run_module(mod, passthrough)

app = main
