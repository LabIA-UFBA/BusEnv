#!/usr/bin/env python3
import os, re, json
import numpy as np
import pandas as pd
from collections import defaultdict

BASE_DIR = "/mnt/ssd1/wesley/BusEnv/exp_results/mappo_mlp_sunt_bus"
OUT_AGG = "tabela_aggregada_robusta.csv"
OUT_DETAILED = "tabela_detalhada_runs.csv"

# Mapeia o label pelo nome da pasta ou parte do nome
TARGET_LABELS_BY_NAME = {
    "all-equal": "MAPPO-all-equal",
}

FLOAT_TOL = 1e-6

def find_config_tuple_from_name(name: str):
    m = re.search(r"\(([^)]+)\)", name)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split(",")]
    if len(parts) != 4:
        return None
    try:
        return tuple(float(x) for x in parts)
    except:
        return None

def read_progress_csv(path):
    df = pd.read_csv(path)
    reward_candidates = [
        "episode_reward_mean", "episode_return_mean", "rollout/episode_reward_mean",
        "evaluation/episode_reward_mean", "train/mean_reward", "mean_reward"
    ]
    ycol = next((c for c in reward_candidates if c in df.columns), None)
    if ycol is None:
        ycol = next((c for c in df.columns if "reward" in c.lower() or "return" in c.lower()), None)
    if ycol is None:
        raise ValueError(f"Nenhuma coluna de reward encontrada em {path}")

    x_candidates = ["timesteps_total", "env_steps_sampled", "env_steps", "training_iteration", "episodes_total"]
    xcol = next((c for c in x_candidates if c in df.columns), None)

    xs = df[xcol].to_numpy(dtype=float) if xcol is not None else np.arange(len(df), dtype=float)
    ys = df[ycol].to_numpy(dtype=float)
    return xs, ys, ycol, xcol

def metrics_from_series(xs, ys, tail_frac=0.2):
    xs, ys = np.array(xs, dtype=float), np.array(ys, dtype=float)
    mask = ~np.isnan(ys)
    xs, ys = xs[mask], ys[mask]
    if len(ys) == 0:
        raise ValueError("Série vazia/NaN")
    n, k = len(ys), max(1, int(len(ys) * tail_frac))
    final_mean = float(np.mean(ys[-k:]))
    y_min, y_max = float(np.nanmin(ys)), float(np.nanmax(ys))
    if np.isclose(y_max, y_min):
        auc = 0.0
    else:
        ys_norm = (ys - y_min) / (y_max - y_min)
        xs_clean = xs if not np.isclose(xs.max(), xs.min()) else np.arange(len(ys), dtype=float)
        x_span = float(xs_clean.max() - xs_clean.min()) if not np.isclose(xs_clean.max(), xs_clean.min()) else 1.0
        auc = float(np.trapz(ys_norm, xs_clean) / x_span)
        if np.isnan(auc) or np.isinf(auc):
            auc = 0.0
    return final_mean, auc

def collect_runs(base_dir):
    runs_list = []
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base dir não encontrado: {base_dir}")
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if not os.path.isdir(full):
            continue
        cfg = find_config_tuple_from_name(name)
        if cfg is None:
            continue
        algo_from_params = None
        params_path = os.path.join(full, "params.json")
        if os.path.exists(params_path):
            try:
                with open(params_path, "r") as f:
                    pj = json.load(f)
                algo_from_params = pj.get("algorithm") or pj.get("model", {}).get("custom_model_config", {}).get("algorithm")
            except:
                pass
        # define o label pelo nome da pasta
        label = None
        lower_name = name.lower()
        for k, v in TARGET_LABELS_BY_NAME.items():
            if k in lower_name:
                label = v
                break
        runs_list.append({
            "algo": (algo_from_params or "MAPPO").upper(),
            "label": label or "UNKNOWN",
            "cfg": cfg,
            "name": name,
            "path": full,
            "params_path": params_path if os.path.exists(params_path) else None
        })
    return runs_list

if __name__ == "__main__":
    runs = collect_runs(BASE_DIR)

    detailed_rows = []
    for r in runs:
        prog = os.path.join(r["path"], "progress.csv")
        if not os.path.exists(prog):
            print(f"⚠️ run sem progress.csv: {r['name']}")
            continue
        try:
            xs, ys, ycol, xcol = read_progress_csv(prog)
            final_run, auc_run = metrics_from_series(xs, ys, tail_frac=0.2)
        except Exception as e:
            print(f"⚠️ erro lendo {prog}: {e}")
            continue
        detailed_rows.append({
            "Algorithm": r["algo"],
            "Label": r["label"],
            "Config": str(r["cfg"]),
            "Run name": r["name"],
            "Seed": None,
            "progress.csv": prog,
            "reward_column": ycol,
            "x_column": xcol,
            "final_mean_run": final_run,
            "auc_run": auc_run
        })

    df_det = pd.DataFrame(detailed_rows)
    df_det.to_csv(OUT_DETAILED, index=False)
    print(f"Detalhado por run salvo em: {OUT_DETAILED}")

    # agregação
    agg_rows = []
    for (algo, label), group in df_det.groupby(["Algorithm", "Label"]):
        finals = group["final_mean_run"].to_numpy(dtype=float)
        aucs = group["auc_run"].to_numpy(dtype=float)
        N = len(finals)
        mean_final = float(np.mean(finals))
        std_final = float(np.std(finals, ddof=1)) if N > 1 else 0.0
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs, ddof=1)) if N > 1 else 0.0
        agg_rows.append({
            "Algorithm": algo,
            "Label": label,
            "Config (occ,up,sync,energy)": str(group["Config"].iloc[0]),
            "Num runs": N,
            "Final Reward (mean ± std)": f"{mean_final:.3f} ± {std_final:.3f}",
            "AUC (mean ± std)": f"{mean_auc:.3f} ± {std_auc:.3f}"
        })

    df_agg = pd.DataFrame(agg_rows)
    df_agg = df_agg.sort_values(["Label", "Algorithm"]).reset_index(drop=True)
    df_agg.to_csv(OUT_AGG, index=False)
    print(f"Agregado salvo em: {OUT_AGG}")
    print(df_agg.to_string(index=False))
