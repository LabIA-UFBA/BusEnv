"""
Grade compacta de curvas de aprendizado - um subplot por ALGORITMO,
com uma linha por CONTEXTO de treino dentro de cada subplot.

A figura é desenhada já no tamanho FINAL do paper (em polegadas), então
os tamanhos de fonte abaixo são os tamanhos reais, em pontos, que vão
aparecer no PDF. Inclua a figura com width=\\textwidth (figure*) ou
width=\\columnwidth, sem redimensionar.
"""

import os
import glob

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ==========================================================
# CONFIGURAÇÃO DOS DADOS
# ==========================================================

REWARD_COLUMN = "episode_reward_mean"
X_COLUMN = "timesteps_total"

MAX_TIMESTEPS = 400000    # ajuste conforme o tamanho real do seu treino
GRID_POINTS = 300
SMOOTH_SIGMA = 3.0

# pastas dentro de runs_lexicografico que NÃO são contextos
IGNORED_PREFIXES = ("graficos",)

ALGORITHMS = [
    "COMA", "HAPPO", "HATRPO",
    "IA2C", "IPPO", "ITRPO",
    "MAA2C", "MAPPO", "MATRPO"
]

LETTERS = "abcdefghijklmnopqrstuvwxyz"


# ==========================================================
# LAYOUT DA GRADE
# ==========================================================

# linhas x colunas. Precisa ter pelo menos len(ALGORITHMS) células.
# 2 x 5 = 10 células: 9 algoritmos + 1 célula livre para a legenda.
N_ROWS = 2
N_COLS = 5

# True  -> legenda dentro da célula livre da grade (economiza espaço)
# False -> legenda numa faixa abaixo da grade
# (se não houver célula livre, a legenda vai abaixo automaticamente)
LEGEND_IN_GRID = True

# Tamanho FINAL da figura no paper, em polegadas.
# figure* em duas colunas ~ 7.0-7.2 in de largura; coluna única ~ 3.3 in.
FIG_SIZE = (7.2, 3.9)
FIG_DPI = 300

SHOW_SUPTITLE = False     # no paper o título fica na legenda (caption)


# ==========================================================
# TAMANHOS DE FONTE (pontos reais no tamanho final)
# ==========================================================

FONT_SUPTITLE = 10
FONT_TITLE = 9
FONT_AXIS_LABEL = 8
FONT_TICKS = 7
FONT_LEGEND = 7.5

LINE_WIDTH = 1.4
LEGEND_LINE_WIDTH = 2.6


# ==========================================================
# RÓTULOS CURTOS DOS CONTEXTOS (legenda)
# Pastas que não estiverem aqui aparecem com o próprio nome.
# A ordem desta lista também define a ordem na legenda.
# ==========================================================

CONTEXT_LABEL = {
    "HIBRID": "CHiP",
    "RUN_ENERGY_MAX": "EFF-MAX",
    "RUN_ENERGY_MAX_WEIGHT": "EFF-MAX-W",
    "RUN_OCC_MAX": "OCC-MAX",
    "RUN_OCC_MAX_WEIGHT": "OCC-MAX-W",
    "RUN_SYNC_MAX": "SYNC-MAX",
    "RUN_SYNC_MAX_WEIGHT": "SYNC-MAX-W",
    "RUN_UPTIME_MAX": "UPTIME-MAX",
    "RUN_UPTIME_MAX_WEIGHT": "UPTIME-MAX-W",
    "WEIGHT": "WEIGHT",
}


# ==========================================================
# ESTILO (dataviz skill, light mode)
# ==========================================================

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

# uma cor fixa por contexto, reaproveitada em todos os subplots
CONTEXT_PALETTE = [
    "#2a78d6", "#eb6834", "#1baf7a", "#c94f7c", "#8a5fd6",
    "#d6b32a", "#4fa6c9", "#c94f4f", "#6fae3f", "#9c6b3f",
]


# ==========================================================
# CAMINHOS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

RUNS_DIR = os.path.join(PROJECT_ROOT, "runs_lexicografico")

OUTPUT_DIR = os.path.join(RUNS_DIR, "graficos_contextos")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# DESCOBRIR OS CONTEXTOS DE TREINO
# ==========================================================

def find_context_dirs(runs_dir):

    contexts = []

    for entry in sorted(os.listdir(runs_dir)):

        full_path = os.path.join(runs_dir, entry)

        if not os.path.isdir(full_path):
            continue

        if entry.lower().startswith(IGNORED_PREFIXES):
            continue

        contexts.append(entry)

    # Ordena conforme CONTEXT_LABEL; pastas desconhecidas vão ao final
    order = list(CONTEXT_LABEL.keys())

    contexts.sort(
        key=lambda c: order.index(c) if c in order else len(order)
    )

    return contexts


CONTEXTS = find_context_dirs(RUNS_DIR)

CONTEXT_COLOR = {
    context: CONTEXT_PALETTE[i % len(CONTEXT_PALETTE)]
    for i, context in enumerate(CONTEXTS)
}

print(f"Contextos encontrados: {CONTEXTS}")


# ==========================================================
# Encontrar progress.csv de um algoritmo dentro de um contexto
# ==========================================================

def find_progress_csvs(context_dir, algorithm):

    pattern = os.path.join(
        context_dir,
        f"{algorithm}_*",
        "**",
        "progress.csv"
    )

    return sorted(glob.glob(pattern, recursive=True))


# ==========================================================
# Carregar as curvas (timesteps, reward) de todas as seeds de
# um algoritmo em um contexto
# ==========================================================

def load_seed_curves(context, algorithm):

    context_dir = os.path.join(RUNS_DIR, context)

    csvs = find_progress_csvs(context_dir, algorithm)

    curves = []

    for csv_path in csvs:

        df = pd.read_csv(csv_path)

        if (
            REWARD_COLUMN not in df.columns
            or X_COLUMN not in df.columns
        ):
            continue

        df = (
            df[[X_COLUMN, REWARD_COLUMN]]
            .dropna()
            .sort_values(X_COLUMN)
            .drop_duplicates(subset=X_COLUMN)
        )

        if len(df) < 2:
            continue

        curves.append((
            df[X_COLUMN].to_numpy(dtype=float),
            df[REWARD_COLUMN].to_numpy(dtype=float)
        ))

    return curves


# ==========================================================
# Média entre seeds, numa grade comum de timesteps
# ==========================================================

def aggregate_seeds(curves, grid):

    stacked = np.full((len(curves), len(grid)), np.nan)

    for i, (x, y) in enumerate(curves):
        stacked[i] = np.interp(grid, x, y, left=np.nan, right=np.nan)

    return np.nanmean(stacked, axis=0)


# ==========================================================
# Formatador do eixo X
# ==========================================================

def format_x(x, pos):
    if x == 0:
        return "0"
    return f"{x / 1e3:.0f}k"


# ==========================================================
# Para UM algoritmo: curvas normalizadas de cada contexto,
# numa grade comum de timesteps (só o intervalo em que todos
# os contextos têm dado, sem extrapolar)
# ==========================================================

def compute_normalized_curves_for_algorithm(algorithm):

    per_context_curves = {}
    min_ts = []
    max_ts = []

    for context in CONTEXTS:

        curves = load_seed_curves(context, algorithm)

        if not curves:
            continue

        per_context_curves[context] = curves

        min_ts.append(max(x.min() for x, _ in curves))
        max_ts.append(min(x.max() for x, _ in curves))

    if not per_context_curves:
        return None, {}

    grid_min = max(min_ts)
    grid_max = min(min(max_ts), MAX_TIMESTEPS)

    if grid_max <= grid_min:
        print(f"[AVISO] Intervalo inválido para {algorithm}")
        return None, {}

    grid = np.linspace(grid_min, grid_max, GRID_POINTS)

    normalized = {}

    for context, curves in per_context_curves.items():

        mean_curve = aggregate_seeds(curves, grid)

        smooth_curve = gaussian_filter1d(mean_curve, sigma=SMOOTH_SIGMA)

        vmin = np.nanmin(smooth_curve)
        vmax = np.nanmax(smooth_curve)
        denom = (vmax - vmin) if (vmax - vmin) > 1e-12 else 1.0

        normalized[context] = (smooth_curve - vmin) / denom

    return grid, normalized


# ==========================================================
# MONTAR A GRADE
# ==========================================================

n_algos = len(ALGORITHMS)
n_cells = N_ROWS * N_COLS

if n_algos > n_cells:
    raise ValueError(
        f"A grade {N_ROWS}x{N_COLS} tem {n_cells} células, mas há "
        f"{n_algos} algoritmos. Aumente N_ROWS/N_COLS ou remova "
        f"algoritmos da lista ALGORITHMS."
    )

legend_in_grid = LEGEND_IN_GRID and n_algos < n_cells

fig, axes = plt.subplots(
    N_ROWS, N_COLS,
    figsize=FIG_SIZE,
    dpi=FIG_DPI,
    sharey=True,
    squeeze=False
)

fig.patch.set_facecolor(SURFACE)

axes_flat = axes.flatten()

legend_handles = {}

for idx, algorithm in enumerate(ALGORITHMS):

    ax = axes_flat[idx]
    ax.set_facecolor(SURFACE)

    letter = LETTERS[idx]

    grid, normalized = compute_normalized_curves_for_algorithm(algorithm)

    if grid is None:

        ax.text(
            0.5, 0.5, "sem dados",
            ha="center", va="center",
            fontsize=FONT_AXIS_LABEL, color=INK_MUTED,
            transform=ax.transAxes
        )

    else:

        for context, curve in normalized.items():

            color = CONTEXT_COLOR[context]

            line, = ax.plot(
                grid, curve,
                color=color,
                linewidth=LINE_WIDTH,
                solid_capstyle="round"
            )

            legend_handles.setdefault(context, line)

    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.5, 1])
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_x))

    # Rótulo do eixo X só nas células sem outro gráfico logo abaixo
    if idx + N_COLS >= n_algos:
        ax.set_xlabel("Step", fontsize=FONT_AXIS_LABEL, color=INK_SECONDARY)

    # Rótulo do eixo Y só na primeira coluna (eixo Y é compartilhado)
    if idx % N_COLS == 0:
        ax.set_ylabel(
            "Normalized Reward",
            fontsize=FONT_AXIS_LABEL,
            color=INK_SECONDARY
        )

    ax.tick_params(
        axis="both",
        labelsize=FONT_TICKS,
        colors=INK_SECONDARY,
        length=2.5,
        pad=2
    )

    ax.grid(True, color=GRIDLINE, linewidth=0.6)
    ax.set_axisbelow(True)

    for spine_name, spine in ax.spines.items():
        if spine_name in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_color(BASELINE_AXIS)

    ax.set_title(
        f"({letter}) {algorithm}",
        fontsize=FONT_TITLE,
        color=INK_PRIMARY,
        fontweight="bold",
        pad=4
    )


# Células que sobraram: ocultas (a primeira pode receber a legenda)
for ax in axes_flat[n_algos:]:
    ax.axis("off")


# ==========================================================
# LEGENDA COMPARTILHADA (uma entrada por contexto, para a
# figura toda, em vez de repetir em cada subplot)
# ==========================================================

legend_labels = [
    CONTEXT_LABEL.get(context, context)
    for context in legend_handles.keys()
]

if legend_in_grid:

    leg = axes_flat[n_algos].legend(
        list(legend_handles.values()),
        legend_labels,
        loc="center",
        ncol=1,
        frameon=False,
        fontsize=FONT_LEGEND,
        labelcolor=INK_SECONDARY,
        handlelength=1.8,
        labelspacing=0.35,
        borderaxespad=0
    )

    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.7)

else:

    leg = fig.legend(
        list(legend_handles.values()),
        legend_labels,
        loc="lower center",
        ncol=min(5, max(1, len(legend_handles))),
        frameon=False,
        fontsize=FONT_LEGEND,
        labelcolor=INK_SECONDARY,
        handlelength=2.0,
        columnspacing=1.2,
        labelspacing=0.4,
        bbox_to_anchor=(0.5, 0.0)
    )

    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.7, rect=[0, 0.12, 1, 1])

# Traço da legenda mais grosso, para as cores serem distinguíveis
for legend_line in leg.get_lines():
    legend_line.set_linewidth(LEGEND_LINE_WIDTH)

if SHOW_SUPTITLE:
    fig.suptitle(
        "Training curves normalized by context and algorithm",
        fontsize=FONT_SUPTITLE, color=INK_PRIMARY
    )


# ==========================================================
# SALVAR (PNG para visualizar, PDF vetorial para o paper)
# ==========================================================

base_name = "grade_algoritmos_contextos_compacta"

for ext in ("png", "pdf"):

    output_path = os.path.join(OUTPUT_DIR, f"{base_name}.{ext}")

    fig.savefig(
        output_path,
        facecolor=SURFACE,
        bbox_inches="tight",
        pad_inches=0.02
    )

    print(f"Imagem salva em:\n{output_path}")

plt.close(fig)