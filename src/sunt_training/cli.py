import argparse, importlib, runpy, sys

def _run_module(mod_path: str) -> int:
    try:
        mod = importlib.import_module(mod_path)
        if hasattr(mod, "main"):
            return int(mod.main() or 0)
        # fall back: run module's top-level code (not ideal, but preserves behavior)
        runpy.run_module(mod_path, run_name="__main__")
        return 0
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1

def main(argv=None):
    p = argparse.ArgumentParser(prog="graphx", description="Graph Exploration CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("env-sunt", help="Executa entrypoint do ambiente SUNT (se houver)")       .add_argument("--", nargs=argparse.REMAINDER)

    sub.add_parser("train", help="Treino RLlib").add_argument("--", nargs=argparse.REMAINDER)
    sub.add_parser("obs", help="Geração de observações").add_argument("--", nargs=argparse.REMAINDER)
    sub.add_parser("routes", help="Geração de rotas reais").add_argument("--", nargs=argparse.REMAINDER)
    sub.add_parser("stats", help="Cálculo de estatísticas/médias").add_argument("--", nargs=argparse.REMAINDER)
    sub.add_parser("look-amount", help="Ferramenta lookAmount").add_argument("--", nargs=argparse.REMAINDER)
    sub.add_parser("pkl-medias", help="Ferramenta pklFilesMedias").add_argument("--", nargs=argparse.REMAINDER)
    sub.add_parser("see-routes", help="Ferramenta seeRotsFiles").add_argument("--", nargs=argparse.REMAINDER)
    sub.add_parser("view-pkl", help="Ferramenta viewPklFiles").add_argument("--", nargs=argparse.REMAINDER)
    sub.add_parser("view-graph", help="Visualização de grafos").add_argument("--", nargs=argparse.REMAINDER)

    args = p.parse_args(argv)

    # Map command -> module
    modmap = {
        "env-sunt": "sunt_training.envs.sunt_env",
        "train": "sunt_training.pipelines.train_rllib",
        "obs": "sunt_training.pipelines.observations",
        "routes": "sunt_training.pipelines.real_routes",
        "stats": "sunt_training.pipelines.stats",
        "look-amount": "sunt_training.tools.look_amount",
        "pkl-medias": "sunt_training.tools.pkl_medias",
        "see-routes": "sunt_training.tools.see_routes",
        "view-pkl": "sunt_training.tools.view_pkl",
        "view-graph": "sunt_training.viz.view_graph",
    }

    mod = modmap[args.cmd]
    # Pass-through args after '--' if present
    passthrough = []
    if getattr(args, "_", None):
        passthrough = args._

    sys.argv = [mod] + passthrough
    return _run_module(mod)

app = main
