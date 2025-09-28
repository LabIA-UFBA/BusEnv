import argparse
import runpy
import sys
from typing import List  # <- Import necessário para Python 3.8

def _run_module(mod_path: str, args: List[str]) -> int:
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
    p = argparse.ArgumentParser(prog="marllib", description="Graph Exploration CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Each command captures additional arguments after '--'
    sub.add_parser("env-sunt", help="Run the entrypoint of the SUNT environment") \
        .add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("train", help="Train with RLlib") \
        .add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("train-marllib-a2c", help="Train with MARLlib A2C (default)") \
        .add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("train-custom-a2c", help="Train with MARLlib custom A2C") \
        .add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("stats", help="Calculate dataset statistics/averages") \
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
    sub.add_parser("view-especific-node", help="Tool viewEspecificNode") \
        .add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("view-metrics", help="Visualize training metrics") \
        .add_argument("args", nargs=argparse.REMAINDER)

    args = p.parse_args(argv)

    # Command mapping -> module
    modmap = {
        "env-sunt": "src.envs.sunt_env",
        "train": "src.pipelines.train_rllib",
        "train-marllib-a2c": "src.pipelines.train_marllib_a2c",
        "train-custom-a2c": "src.pipelines.train_custom_a2c",
        "stats": "src.pipelines.show_dataset_stats",
        "look-amount": "src.tools.look_amount",
        "pkl-medias": "src.tools.pkl_medias",
        "see-routes": "src.tools.see_routes",
        "view-pkl": "src.tools.view_pkl",
        "view-graph": "src.viz.view_graph",
        "view-especific-node": "src.tools.view_especific_nodePkl",
        "view-metrics": "src.tools.view_metrics",
    }

    mod = modmap[args.cmd]
    passthrough = getattr(args, "args", []) or []

    # Run the mapped module with passthrough args
    return _run_module(mod, passthrough)

def app():
    return main()
