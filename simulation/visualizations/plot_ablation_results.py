"""Generate publication-quality ablation study figures and tables.

Reads the multi-seed ablation summary CSV and produces:
1. A single-factor bar chart with error bars (Group 2 amplitude)
2. A full factorial heatmap with component grid
3. A LaTeX summary table
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from pathlib import Path

# ── Style ──────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT_DIR = Path("visualizations")


def load_summary() -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / "ablation_summary.csv")


# ── Figure 1: Single-factor bar chart with error bars ─────────────────────────
def plot_single_factor_bars(df: pd.DataFrame):
    """Horizontal bar chart of main single-factor ablations."""
    keys = [
        "baseline",
        "no_homing",
        "no_no_turning_noise",
        "homogenize_init",
        "no_walls",
        "no_external_driver",
        "no_speed_transfer",
    ]
    labels = [
        "Baseline\n(all components)",
        "No homing\n(location preference)",
        "No turning noise\n(deterministic)",
        "Homogenized\ninitial positions",
        "No walls\n(unbounded arena)",
        "No external driver",
        "No speed transfer",
    ]
    sub = df[df["label"].isin(keys)].copy()
    sub["order"] = sub["label"].map({k: i for i, k in enumerate(keys)})
    sub = sub.sort_values("order")

    amp = sub["amplitude_group2_mean"].values
    err = sub["amplitude_group2_sem"].values
    sig = sub["rhythmicity_p_value_group2_mean"].values < 0.05

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    colors = ["#2196F3" if s else "#BDBDBD" for s in sig]
    bars = ax.barh(range(len(labels)), amp, xerr=err, color=colors,
                   height=0.6, edgecolor="white", linewidth=0.5,
                   capsize=3, error_kw={"linewidth": 0.8, "color": "#555"})

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Group 2 rhythmicity amplitude (mean ± SEM, n=5)")
    ax.set_title("Effect of ablating individual model components")
    ax.invert_yaxis()

    # Value annotations
    for i, (v, e, s) in enumerate(zip(amp, err, sig)):
        txt = f"{v:.3f}" if v > 0.005 else f"{v:.4f}"
        sig_txt = " *" if s else " n.s."
        ax.text(v + e + 0.005, i, txt + sig_txt, va="center", fontsize=7.5,
                color="#333")

    ax.set_xlim(0, max(amp + err) * 1.35)

    legend_elements = [
        Patch(facecolor="#2196F3", label="Rhythmic (p < 0.05)"),
        Patch(facecolor="#BDBDBD", label="Not rhythmic"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "ablation_single_factor.pdf")
    fig.savefig(OUT_DIR / "ablation_single_factor.png")
    print(f"Saved: {OUT_DIR}/ablation_single_factor.pdf")
    plt.close(fig)


# ── Figure 2: Factorial heatmap ───────────────────────────────────────────────
def plot_factorial_heatmap(df: pd.DataFrame):
    """Heatmap for 4-factor factorial ablation."""
    factors = ["walls", "homing", "speed_transfer", "turning_noise"]
    factor_labels = ["Walls", "Homing", "Speed\ntransfer", "Turning\nnoise"]

    # Filter to factorial conditions only
    factorial_labels = [
        "baseline",
        "no_no_turning_noise",
        "no_speed_transfer",
        "no_speed_transfer+no_turning_noise",
        "no_homing",
        "no_homing+no_turning_noise",
        "no_homing+speed_transfer",
        "no_homing+speed_transfer+no_turning_noise",
        "no_walls",
        "no_walls+no_turning_noise",
        "no_walls+speed_transfer",
        "no_walls+speed_transfer+no_turning_noise",
        "no_walls+homing",
        "no_walls+homing+no_turning_noise",
        "no_walls+homing+speed_transfer",
        "no_walls+homing+speed_transfer+no_turning_noise",
    ]

    sub = df[df["label"].isin(factorial_labels)].copy()

    # Parse which factors are ablated
    rows = []
    for _, r in sub.iterrows():
        label = r["label"]
        ablated = {f: False for f in factors}
        if label != "baseline":
            parts = label.replace("no_no_turning_noise", "no_turning_noise").split("+")
            for p in parts:
                p = p.strip()
                if p == "no_walls":
                    ablated["walls"] = True
                elif p in ("no_homing", "homing"):
                    ablated["homing"] = True
                elif p in ("no_speed_transfer", "speed_transfer"):
                    ablated["speed_transfer"] = True
                elif p in ("no_turning_noise", "turning_noise"):
                    ablated["turning_noise"] = True
        rows.append({
            "label": label,
            **{f"ablated_{f}": ablated[f] for f in factors},
            "amp": r["amplitude_group2_mean"],
            "err": r["amplitude_group2_sem"],
            "sig": r["rhythmicity_p_value_group2_mean"] < 0.05,
        })

    data = pd.DataFrame(rows)
    data = data.sort_values(
        [f"ablated_{f}" for f in factors], ascending=True
    ).reset_index(drop=True)

    n = len(data)
    fig, (ax_left, ax_main) = plt.subplots(
        1, 2, figsize=(7, 4.5),
        gridspec_kw={"width_ratios": [1.2, 3], "wspace": 0.05}
    )

    # Left panel: component grid
    for i, row in data.iterrows():
        for j, f in enumerate(factors):
            color = "#E0E0E0" if row[f"ablated_{f}"] else "#2196F3"
            ax_left.add_patch(plt.Rectangle((j, n - 1 - i), 0.8, 0.8,
                                            facecolor=color, edgecolor="white",
                                            linewidth=1))

    ax_left.set_xlim(-0.1, len(factors))
    ax_left.set_ylim(-0.2, n)
    ax_left.set_xticks([x + 0.4 for x in range(len(factors))])
    ax_left.set_xticklabels(factor_labels, ha="center")
    ax_left.set_yticks([])
    ax_left.spines["left"].set_visible(False)
    ax_left.spines["bottom"].set_visible(False)
    ax_left.tick_params(bottom=False)
    ax_left.set_title("Components", fontsize=9, pad=8)

    # Right panel: bars with error bars
    bar_colors = ["#2196F3" if s else "#BDBDBD" for s in data["sig"]]
    ax_main.barh(
        range(n), data["amp"].values, xerr=data["err"].values,
        color=bar_colors, height=0.7, edgecolor="white", linewidth=0.5,
        capsize=2, error_kw={"linewidth": 0.7, "color": "#555"}
    )
    ax_main.invert_yaxis()
    ax_main.set_yticks([])
    ax_main.set_xlabel("Group 2 amplitude (mean ± SEM, n=5)")
    ax_main.set_title("Rhythmicity of non-rhythmic group", fontsize=9, pad=8)

    for i, (v, e) in enumerate(zip(data["amp"], data["err"])):
        txt = f"{v:.3f}" if v > 0.005 else f"{v:.4f}"
        ax_main.text(v + e + 0.003, i, txt, va="center", fontsize=7, color="#555")

    ax_main.set_xlim(0, data["amp"].max() * 1.35)

    legend_elements = [
        Patch(facecolor="#2196F3", label="Active / Rhythmic (p < 0.05)"),
        Patch(facecolor="#E0E0E0", label="Ablated / Not rhythmic"),
    ]
    ax_main.legend(handles=legend_elements, loc="lower right", frameon=False,
                   fontsize=7.5)

    fig.suptitle("Full factorial ablation study (n=5 seeds per condition)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ablation_factorial_heatmap.pdf")
    fig.savefig(OUT_DIR / "ablation_factorial_heatmap.png")
    print(f"Saved: {OUT_DIR}/ablation_factorial_heatmap.pdf")
    plt.close(fig)


# ── LaTeX table ───────────────────────────────────────────────────────────────
def generate_latex_table(df: pd.DataFrame):
    """LaTeX table of selected ablation conditions."""
    keys = [
        "baseline",
        "no_speed_transfer",
        "no_homing",
        "no_walls",
        "no_no_turning_noise",
        "no_external_driver",
        "homogenize_init",
        "no_homing+speed_transfer",
        "no_walls+speed_transfer",
        "no_walls+homing",
        "no_walls+homing+speed_transfer",
    ]
    names = {
        "baseline": "Baseline (all components)",
        "no_speed_transfer": "No speed transfer",
        "no_homing": "No homing (location preference)",
        "no_walls": "No walls (unbounded arena)",
        "no_no_turning_noise": "No turning noise",
        "no_external_driver": "No external driver",
        "homogenize_init": "Homogenized initial positions",
        "no_homing+speed_transfer": "No homing + no speed transfer",
        "no_walls+speed_transfer": "No walls + no speed transfer",
        "no_walls+homing": "No walls + no homing",
        "no_walls+homing+speed_transfer": "No walls, homing, or speed transfer",
    }

    sub = df[df["label"].isin(keys)].copy()
    sub["order"] = sub["label"].map({k: i for i, k in enumerate(keys)})
    sub = sub.sort_values("order")

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Ablation study: effect of removing individual model components on "
        r"rhythmicity transfer (mean $\pm$ SEM across $n{=}5$ independent seeds). "
        r"$A_1$ and $A_2$ are the fitted sinusoidal amplitudes of the Group~1 "
        r"(forager) and Group~2 (hive bee) mean speed time series, respectively. "
        r"$\Delta\phi$ is the phase lag of Group~2 relative to Group~1 in degrees, "
        r"and $p_2$ is the permutation-test $p$-value for Group~2 rhythmicity "
        r"(200 permutations). Simulations used density factor $d{=}1.0$, "
        r"day duration $T_{\mathrm{day}}{=}100$~steps, and decay rate $\gamma{=}0.8$. "
        r"Speed transfer is the sole essential mechanism; walls provide secondary "
        r"amplification by confining agents and increasing contact rates.\textsuperscript{a}}",
        r"\label{tab:ablation}",
        r"\small",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"\textbf{Condition} & $A_1$ & $A_2$ & $\Delta\phi$ (\textdegree) & $p_2$ \\",
        r"\midrule",
    ]

    for _, row in sub.iterrows():
        name = names.get(row["label"], row["label"])
        a1_m = row["amplitude_group1_mean"]
        a1_e = row["amplitude_group1_sem"]
        a2_m = row["amplitude_group2_mean"]
        a2_e = row["amplitude_group2_sem"]
        ps_m = row["phase_shift_g2_minus_g1_mean"]
        ps_e = row["phase_shift_g2_minus_g1_sem"]
        p2_m = row["rhythmicity_p_value_group2_mean"]

        a1 = f"${a1_m:.2f} \\pm {a1_e:.2f}$"
        if a2_m > 0.005:
            a2 = f"${a2_m:.3f} \\pm {a2_e:.3f}$"
        else:
            a2 = f"${a2_m:.4f} \\pm {a2_e:.4f}$"
        # Omit phase shift when Group 2 rhythm is not significant
        if p2_m < 0.05:
            ps = f"${ps_m:.1f} \\pm {ps_e:.1f}$"
        else:
            ps = r"---"
        if p2_m < 0.01:
            p2 = "$<$0.01"
        else:
            p2 = f"{p2_m:.2f}"

        lines.append(f"  {name} & {a1} & {a2} & {ps} & {p2} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{2pt}",
        r"{\scriptsize \textsuperscript{a}Phase shift omitted where Group~2 rhythmicity was not significant ($p \geq 0.05$).}",
        r"\end{table}",
    ]

    table_tex = "\n".join(lines)
    out_path = OUT_DIR / "ablation_table.tex"
    out_path.write_text(table_tex)
    print(f"Saved: {out_path}")
    print(table_tex)


if __name__ == "__main__":
    plt.switch_backend("Agg")
    df = load_summary()
    print(f"Loaded {len(df)} conditions\n")
    print(df[["label", "amplitude_group2_mean", "amplitude_group2_sem"]].to_string(index=False))

    plot_single_factor_bars(df)
    plot_factorial_heatmap(df)
    generate_latex_table(df)
    print("\nDone.")
