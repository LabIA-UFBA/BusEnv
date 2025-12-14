#!/usr/bin/env python3
import os
import re
import json
import numpy as np
import pandas as pd

BASE_DIR = "/mnt/ssd1/wesley/BusEnv/exp_results/mappo_mlp_sunt_bus"
OUT_AGG = os.path.join(BASE_DIR, "tabela_aggregada_robusta.csv")
OUT_DETAILED = os.path.join(BASE_DIR, "tabela_detalhada_runs.csv")

# Mapeia tuplo de config -> label (ocorrência, uptime, sync, energy)
CONFIG_TO_LABEL = {
    (0.0, 0.0, 0.0, 1.0): "MAPPO-energy_efficiency",
    (1.0, 0.0, 0.0, 0.0): "MAPPO-occ_penalty",
    (0.0, 0.0, 1.0, 0.0): "MAPPO-sync_score",
    (0.0, 1.0, 0.0, 0.0): "MAPPO-uptime_bonus",
}

# também permitir reconhecer pelo nome da pasta (substring)
NAME_LABEL_SUBSTRINGS = {
    "mappo-energy_efficiency": "MAPPO-energy_efficiency",
    "mappo-occ_penalty": "MAPPO-occ_penalty",
    "mappo-sync_score": "MAPPO-sync_score",
    "mappo-uptime_bonus": "MAPPO-uptime_bonus",
}

FLOAT_TOL = 1e-6

def find_config_tuple_from_name(name: str):
    """Procura por '(a,b,c,d)' no nome e retorna tuplo de floats se válido."""
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

def label_for_run(name: str, cfg):
    """Determina label pelo tuplo de cfg (preferido) ou pelo nome (substring)."""
    if cfg is not None:
        # comparar floats com tolerância
        for k, v in CONFIG_TO_LABEL.items():
            if all(abs(a - b) < FLOAT_TOL for a, b in zip(k, cfg)):
                return v
    lower_name = name.lower()
    for substr, v in NAME_LABEL_SUBSTRINGS.items():
        if substr in lower_name:
            return v
    return "UNKNOWN"

def collect_runs(base_dir):
    runs_list = []
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base dir não encontrado: {base_dir}")
    for name in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, name)
        if not os.path.isdir(full):
            continue
        cfg = find_config_tuple_from_name(name)
        algo_from_params = None
        seed = None
        params_path = os.path.join(full, "params.json")
        if os.path.exists(params_path):
            try:
                with open(params_path, "r") as f:
                    pj = json.load(f)
                algo_from_params = pj.get("algorithm") or pj.get("model", {}).get("custom_model_config", {}).get("algorithm")
                # tentativa de extrair seed se houver
                seed = pj.get("seed") or pj.get("experiment_seed") or pj.get("config", {}).get("seed")
            except Exception:
                pass

        label = label_for_run(name, cfg)
        runs_list.append({
            "algo": (algo_from_params or "MAPPO").upper(),
            "label": label,
            "cfg": cfg,
            "name": name,
            "path": full,
            "params_path": params_path if os.path.exists(params_path) else None,
            "seed": seed
        })
    return runs_list

if __name__ == "__main__":
    runs = collect_runs(BASE_DIR)
    if len(runs) == 0:
        print("Nenhuma run encontrada em", BASE_DIR)
        raise SystemExit(1)

    # opcional: filtrar apenas labels conhecidos (os 4 exemplos solicitados)
    target_labels = set(CONFIG_TO_LABEL.values())
    runs = [r for r in runs if r["label"] in target_labels]
    if len(runs) == 0:
        print("Nenhuma run com os labels alvo encontrada. Verifique nomes / tuplos nas pastas.")
        raise SystemExit(1)

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
            "Seed": r.get("seed"),
            "progress.csv": prog,
            "reward_column": ycol,
            "x_column": xcol,
            "final_mean_run": final_run,
            "auc_run": auc_run
        })

    if len(detailed_rows) == 0:
        print("Nenhuma run processada com sucesso.")
        raise SystemExit(1)

    df_det = pd.DataFrame(detailed_rows)
    df_det.to_csv(OUT_DETAILED, index=False)
    print(f"Detalhado por run salvo em: {OUT_DETAILED}")

    # agregação
    agg_rows = []
    for (algo, label), group in df_det.groupby(["Algorithm", "Label"]):
        finals = group["final_mean_run"].to_numpy(dtype=float)
        aucs = group["auc_run"].to_numpy(dtype=float)
        N = len(finals)
        mean_final = float(np.mean(finals)) if N > 0 else 0.0
        std_final = float(np.std(finals, ddof=1)) if N > 1 else 0.0
        mean_auc = float(np.mean(aucs)) if N > 0 else 0.0
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
