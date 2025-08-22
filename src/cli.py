import argparse
import importlib
import runpy
import sys


def _run_module(mod_path: str) -> int:
    try:
        mod = importlib.import_module(mod_path)
        if hasattr(mod, "main"):
            return int(mod.main() or 0)
        # fallback: run module's top-level code
        runpy.run_module(mod_path, run_name="__main__")
        return 0
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1


def main(argv=None):
    p = argparse.ArgumentParser(prog="graphx", description="Graph Exploration CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("env-sunt", help="Executa entrypoint do ambiente SUNT (se houver)") \
        .add_argument("args", nargs=argparse.REMAINDER)  # capture everything after --
    sub.add_parser("train", help="Treino RLlib") \
        .add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("stats", help="Cálculo de estatísticas/médias") \
        .add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("look-amount", help="Ferramenta lookAmount") \
        .add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("pkl-medias", help="Ferramenta pklFilesMedias") \
        .add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("see-routes", help="Ferramenta seeRoutesFiles") \
        .add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("view-pkl", help="Ferramenta viewPklFiles") \
        .add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("view-graph", help="Visualização de grafos") \
        .add_argument("args", nargs=argparse.REMAINDER)

    args = p.parse_args(argv)

    # Map command -> module inside src package
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

    # Pass-through args after '--' if present
    passthrough = getattr(args, "args", []) or []
    sys.argv = [mod] + passthrough

    return _run_module(mod)


app = main
