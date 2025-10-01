import argparse
import runpy
import sys
from typing import List  # <- Import necessário para Python 3.8
from contextlib import nullcontext

# --- Optional CodeCarbon import (only needed when --cc is used) ---
def _make_tracker(enabled: bool, *, project_name: str, output_dir: str = None,
                  offline: bool = False, country_iso_code: str = None,
                  measure_power_secs: int = None, gpu_ids: str = None,
                  log_level: str = "warning"):
    """
    Create a CodeCarbon tracker if enabled; otherwise return a no-op context.
    """
    if not enabled:
        return nullcontext(), None

    try:
        from codecarbon import EmissionsTracker
    except Exception as e:
        # Graceful fallback if CodeCarbon isn't installed
        print(
            "[codecarbon] CodeCarbon is not installed. Install with: pip install codecarbon",
            file=sys.stderr,
        )
        return nullcontext(), None

    tracker = EmissionsTracker(
        project_name=project_name,
        output_dir=output_dir,
        save_to_file=True,
        save_to_api=False,          # flip to True if you use the SaaS API
        measure_power_secs=measure_power_secs,
        gpu_ids=gpu_ids,
        country_iso_code=country_iso_code,  # required in offline mode
        log_level=log_level,
        offline=offline,
    )

    class _TrackerCtx:
        def _enter_(self):
            tracker.start()
            return tracker
        def _exit_(self, exc_type, exc, tb):
            try:
                emissions = tracker.stop()
                if emissions is not None:
                    # Keep a small, human-friendly line in stderr in addition to the CSV/JSON
                    print(
                        f"[codecarbon] {project_name} emissions: {emissions:.6f} kg CO₂eq",
                        file=sys.stderr,
                    )
            except Exception as e:
                print(f"[codecarbon] Failed to stop tracker: {e}", file=sys.stderr)

    return _TrackerCtx(), tracker


def _run_module(mod_path: str, args: List[str]) -> int:
    """
    Run the module as _main_ passing the arguments.
    """
    sys.argv = [mod_path] + args
    try:
        runpy.run_module(mod_path, run_name="_main_")
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

    # ---------------- CodeCarbon flags ----------------
    cc_group = p.add_mutually_exclusive_group()
    cc_group.add_argument("--cc", dest="cc", action="store_true", help="Enable CodeCarbon (default).")
    cc_group.add_argument("--no-cc", dest="cc", action="store_false", help="Disable CodeCarbon.")
    p.set_defaults(cc=True)

    p.add_argument("--cc-output-dir", default=None,
                   help="Directory to store CodeCarbon CSV/JSON (defaults to ./codecarbon).")
    p.add_argument("--cc-offline", action="store_true",
                   help="Run CodeCarbon in offline mode (requires --cc-country).")
    p.add_argument("--cc-country", default=None,
                   help="ISO 3166 alpha-3 country code for energy mix (e.g., 'BRA', 'USA').")
    p.add_argument("--cc-measure-power-secs", type=int, default=None,
                   help="Sampling interval in seconds for power measurement (default CodeCarbon behavior if omitted).")
    p.add_argument("--cc-gpu-ids", default=None,
                   help="Comma-separated GPU ids to monitor (e.g., '0,1').")
    p.add_argument("--cc-log-level", default="warning",
                   choices=["debug", "info", "warning", "error"],
                   help="CodeCarbon internal log level.")
    p.add_argument("--cc-run-id", default=None,
                   help="Optional run identifier (e.g., 'run01', 'expA', etc.). Defaults to timestamp.")

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

    # --- Generate project name for CodeCarbon ---
    import datetime
    run_id = args.cc_run_id or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    project_name = f"marllib:{args.cmd}:{run_id}"

    tracker_ctx, _tracker = _make_tracker(
        enabled=args.cc,
        project_name=project_name,
        output_dir=args.cc_output_dir,
        offline=args.cc_offline,
        country_iso_code=args.cc_country,
        measure_power_secs=args.cc_measure_power_secs,
        gpu_ids=args.cc_gpu_ids,
        log_level=args.cc_log_level,
    )

    with tracker_ctx:
        return _run_module(mod, passthrough)


def app():
    return main()