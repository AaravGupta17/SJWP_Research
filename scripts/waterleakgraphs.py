from __future__ import annotations

from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


NETWORKS = ["Network_3", "Network_5", "Network_6"]
MATERIALS = ["CI", "DI", "PVC", "STEEL"]
DEFAULT_DEMAND = 1.0
DAYS = np.arange(1, 8)


def _resolve_network_root() -> Path:
    # Prefer user-specified relative layout if present.
    here = Path(__file__).resolve()
    candidate = (here.parent / ".." / "data" / "NetworkList").resolve()
    if candidate.exists():
        return candidate
    return (here.parent / ".." / "datasets" / "NetworkList").resolve()


def _parse_demand_from_name(name: str) -> float | None:
    # Expected pattern: leak-<scenario>-<demand>.csv
    parts = name.replace(".csv", "").split("-")
    if len(parts) < 3:
        return None
    try:
        return float(parts[-1])
    except ValueError:
        return None


def _collect_stats(network_root: Path) -> dict[tuple[str, float], dict[str, float]]:
    # Aggregate per (material, demand) across Network_3/5/6 using mean of per-file totals.
    per_net = defaultdict(lambda: defaultdict(list))
    per_net_pipes = defaultdict(lambda: defaultdict(list))

    for network in NETWORKS:
        net_dir = network_root / network
        if not net_dir.exists():
            print(f"Skipping missing network: {net_dir}")
            continue
        for material in MATERIALS:
            mat_dir = net_dir / material
            if not mat_dir.exists():
                print(f"Skipping missing material: {mat_dir}")
                continue
            for csv_path in mat_dir.glob("leak-*-*.csv"):
                demand = _parse_demand_from_name(csv_path.name)
                if demand is None:
                    continue
                try:
                    df = pd.read_csv(csv_path, usecols=["Leak_Flow_Lps"])
                except Exception as exc:
                    print(f"Failed reading {csv_path}: {exc}")
                    continue
                total_leak_flow = float(df["Leak_Flow_Lps"].sum())
                pipes = int(len(df))
                per_net[(network, material)][demand].append(total_leak_flow)
                per_net_pipes[(network, material)][demand].append(pipes)

    aggregated = {}
    for material in MATERIALS:
        for (network, mat), demand_map in per_net.items():
            if mat != material:
                continue
            for demand, flows in demand_map.items():
                if not flows:
                    continue
                mean_flow = float(np.mean(flows))
                mean_pipes = float(np.mean(per_net_pipes[(network, material)][demand]))
                aggregated.setdefault((material, demand), {"flows": [], "pipes": []})
                aggregated[(material, demand)]["flows"].append(mean_flow)
                aggregated[(material, demand)]["pipes"].append(mean_pipes)

    # Final aggregation across networks
    final_stats = {}
    for key, vals in aggregated.items():
        mean_flow = float(np.mean(vals["flows"]))
        mean_pipes = int(round(float(np.mean(vals["pipes"]))))
        final_stats[key] = {
            "mean_flow_lps": mean_flow,
            "pipes": mean_pipes,
            "networks_used": len(vals["flows"]),
        }
    return final_stats


def _cumulative_loss(flow_lps: float) -> np.ndarray:
    liters_per_day = flow_lps * 86400.0
    return liters_per_day * DAYS


def _plot_materials_over_7_days(stats: dict[tuple[str, float], dict[str, float]], out_dir: Path) -> None:
    demand = DEFAULT_DEMAND
    if not any(d == demand for (_, d) in stats.keys()):
        demand = sorted({d for (_, d) in stats.keys()})[0]

    plt.figure(figsize=(10, 6))
    for material in MATERIALS:
        key = (material, demand)
        if key not in stats:
            continue
        flow = stats[key]["mean_flow_lps"]
        pipes = stats[key]["pipes"]
        label = f"{material} | demand {demand:.1f} | pipes {pipes}"
        plt.plot(DAYS, _cumulative_loss(flow), marker="o", label=label)

    plt.title(f"Cumulative Water Loss Over 7 Days (Demand {demand:.1f})")
    plt.xlabel("Day")
    plt.ylabel("Cumulative Water Loss (Liters)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    out_path = out_dir / "water_loss_materials_7days.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_demand_material_sorted(stats: dict[tuple[str, float], dict[str, float]], out_dir: Path) -> None:
    demands = sorted({d for (_, d) in stats.keys()})
    material_colors = {
        "CI": "#1f77b4",
        "DI": "#ff7f0e",
        "PVC": "#2ca02c",
        "STEEL": "#d62728",
    }
    demand_styles = {
        0.7: "-",
        1.0: "--",
        1.2: ":",
    }

    plt.figure(figsize=(12, 7))
    for demand in demands:
        for material in MATERIALS:
            key = (material, demand)
            if key not in stats:
                continue
            flow = stats[key]["mean_flow_lps"]
            pipes = stats[key]["pipes"]
            label = f"demand {demand:.1f} | {material} | pipes {pipes}"
            plt.plot(
                DAYS,
                _cumulative_loss(flow),
                color=material_colors.get(material, None),
                linestyle=demand_styles.get(demand, "-"),
                linewidth=2,
                label=label,
            )

    plt.title("Cumulative Water Loss Over 7 Days (Sorted by Demand, Material)")
    plt.xlabel("Day")
    plt.ylabel("Cumulative Water Loss (Liters)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, ncol=2)
    out_path = out_dir / "water_loss_demand_material_7days.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    network_root = _resolve_network_root()
    out_dir = (Path(__file__).resolve().parent / ".." / "plots").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Network root: {network_root}")
    stats = _collect_stats(network_root)
    if not stats:
        print("No leak CSVs found. Check paths and file names.")
        return

    _plot_materials_over_7_days(stats, out_dir)
    _plot_demand_material_sorted(stats, out_dir)

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
