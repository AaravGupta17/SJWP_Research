import wntr
import numpy as np
import pandas as pd
import random
import os
import math
from tqdm import tqdm
from wntr.epanet.io import InpFile
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ======================================================
# CONFIGURATION
# ======================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INP_FILE = os.path.join(SCRIPT_DIR, "../inp/mend.inp")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../datasets/NetworkList/Network_9")
TEMPLATE_DIR = os.path.join(OUTPUT_DIR, "templates")

MATERIAL = "PVC"
ACOUSTIC_SPEED = 395.0
ATTENUATION_ALPHA = 0.008

DEMAND_LEVELS = [0.0, 0.18, 0.47]
DEMAND_WEIGHTS = {"J_OL": 0.45, "J_LC": 0.25, "J_CC": 0.20, "J_GL": 0.10}

SENSOR_POSITIONS = [11.75, 35.25]
SENSOR_NODES = ["J_S1", "J_S2"]

LEAK_TYPES = [
    {"name": "Orifice",              "weight": 0.40, "Cd": 0.61, "area_m2": 2.01e-6},
    {"name": "Longitudinal_Crack",   "weight": 0.25, "Cd": 0.50, "area_m2": 2.00e-6},
    {"name": "Circumferential_Crack","weight": 0.20, "Cd": 0.50, "area_m2": 2.00e-6},
    {"name": "Gasket",               "weight": 0.15, "Cd": 0.45, "area_m2": 2.00e-6},
]

LEAK_POSITIONS = [4.7, 9.4, 14.1, 18.8, 23.5, 28.2, 32.9, 37.6, 42.3]

TOTAL_LEAK_SCENARIOS = 50000
TOTAL_NOLEAK_SCENARIOS = 50000

SIM_DURATION_SEC = 60

# Linear topology: J0 -> J_S1 -> J_OL -> J_LC -> J_CC -> J_GL -> J_S2 -> J_END
NODE_CHAIN = ["J0", "J_S1", "J_OL", "J_LC", "J_CC", "J_GL", "J_S2", "J_END"]
RESERVOIR_PIPES = ["P_RIN", "P_ROUT"]

random.seed(42)
np.random.seed(42)

print = tqdm.write

# ======================================================
# HELPERS
# ======================================================

def emitter_coefficient_wntr(Cd, area_m2):
    """Compute WNTR emitter coefficient (m^3/s per m^0.5)."""
    g = 9.81
    return Cd * area_m2 * math.sqrt(2 * g)


def build_cumulative_map(wn):
    segments = []
    cum = 0.0
    for i in range(len(NODE_CHAIN) - 1):
        n1, n2 = NODE_CHAIN[i], NODE_CHAIN[i + 1]
        for pid, pipe in wn.pipes():
            if pipe.start_node_name == n1 and pipe.end_node_name == n2:
                segments.append((pid, n1, n2, cum, cum + pipe.length))
                cum += pipe.length
                break
            elif pipe.start_node_name == n2 and pipe.end_node_name == n1:
                segments.append((pid, n2, n1, cum, cum + pipe.length))
                cum += pipe.length
                break
    return segments


def get_pipe_at_position(segments, pos):
    for pid, n1, n2, cstart, cend in segments:
        if cstart <= pos <= cend:
            return pid, n1, n2, pos - cstart
    return None, None, None, None


# ======================================================
# TEMPLATE CREATION
# ======================================================

def create_templates():
    os.makedirs(TEMPLATE_DIR, exist_ok=True)

    for pos in LEAK_POSITIONS:
        pos_label = f"pos_{pos}".replace(".", "_")
        tmpl = os.path.join(TEMPLATE_DIR, f"{pos_label}.inp")
        if os.path.exists(tmpl):
            continue

        wn = wntr.network.WaterNetworkModel(INP_FILE)
        segments = build_cumulative_map(wn)
        pid, n1, n2, local_dist = get_pipe_at_position(segments, pos)

        if pid is None:
            raise ValueError(f"Position {pos}m not on any pipe")

        pipe = wn.get_link(pid)
        orig_len = pipe.length
        leak_node = f"LEAK_{pos_label}"

        wn.add_junction(leak_node, 0.0)
        wn.get_node(leak_node).emitter_coefficient = 0.0

        wn.remove_link(pid)
        wn.add_pipe(f"{pid}_L", n1, leak_node, local_dist, pipe.diameter, pipe.roughness)
        wn.add_pipe(f"{pid}_R", leak_node, n2, orig_len - local_dist, pipe.diameter, pipe.roughness)

        InpFile().write(tmpl, wn)
        print(f"  Created template: {tmpl}")


# ======================================================
# SIMULATION
# ======================================================

def setup_demands(wn, demand_level):
    if demand_level == 0:
        for nd in DEMAND_WEIGHTS:
            wn.get_node(nd).base_demand = 0.0
    else:
        for nd, w in DEMAND_WEIGHTS.items():
            wn.get_node(nd).base_demand = demand_level * w


def run_sim(wn, demand_level):
    setup_demands(wn, demand_level)
    wn.options.time.duration = SIM_DURATION_SEC
    sim = wntr.sim.EpanetSimulator(wn)
    res = sim.run_sim()
    return (
        res.node["pressure"].mean(),
        res.link["flowrate"].mean(),
        res.node["demand"].mean(),
    )


def make_sensor_str(p_s1, p_s2):
    return f"{p_s1:.6f},{p_s2:.6f}"


# ======================================================
# SCENARIO GENERATORS
# ======================================================

def gen_no_leak_rows(demand_level, count):
    wn = wntr.network.WaterNetworkModel(INP_FILE)
    avg_p, avg_q, _ = run_sim(wn, demand_level)
    p_s1 = avg_p.get("J_S1", 0)
    p_s2 = avg_p.get("J_S2", 0)
    sp_str = make_sensor_str(p_s1, p_s2)
    ss_str = f"{SENSOR_POSITIONS[0]:.3f},{SENSOR_POSITIONS[1]:.3f}"

    pipe_rows = []
    for pid, pipe in wn.pipes():
        if pid in RESERVOIR_PIPES:
            continue
        Q = avg_q.get(pid, 0)
        area = math.pi * (pipe.diameter ** 2) / 4
        vel = Q / area if area > 0 else 0
        pipe_rows.append({
            "Pipe_ID": pid, "Pipe_Length_m": pipe.length,
            "Pipe_Diameter_m": pipe.diameter, "Pipe_Roughness": pipe.roughness,
            "Pipe_Material": MATERIAL,
            "Sensor_Positions_m": ss_str,
            "All_Sensor_Pressures_m": sp_str,
            "Avg_Flow_Velocity_mps": vel,
            "Avg_Flow_Rate_lps": Q * 1000,
            "Acoustic_Propagation_Speed_mps": ACOUSTIC_SPEED,
            "Attenuation_Alpha_per_m": ATTENUATION_ALPHA,
            "Leak_Status": 0, "Number_of_Leaks": 0, "Leak_Type": "",
            "Leak_Area_m2": 0.0, "Leak_Flow_Lps": 0.0,
            "Avg_Pressure_at_Leak": None, "True_Leak_Position_m": None,
            "Leak_Distance_2nd_Left_m": 0, "Leak_Distance_Left_m": 0,
            "Leak_Distance_Right_m": 0, "Leak_Distance_2nd_Right_m": 0,
            "Pipe_Pressure_Left1_m": 0, "Pipe_Pressure_Left2_m": 0,
            "Pipe_Pressure_Right1_m": 0, "Pipe_Pressure_Right2_m": 0,
            "Demand_Level": demand_level,
        })

    if count <= len(pipe_rows):
        return pipe_rows[:count]
    repeats = (count // len(pipe_rows)) + 1
    expanded = (pipe_rows * repeats)[:count]
    return expanded


def gen_leak_row(pos, demand_level, leak_type_name, scenario_id, template_path):
    pos_label = f"pos_{pos}".replace(".", "_")
    leak_node = f"LEAK_{pos_label}"

    wn = wntr.network.WaterNetworkModel(template_path)

    lt = next(t for t in LEAK_TYPES if t["name"] == leak_type_name)
    ec = emitter_coefficient_wntr(lt["Cd"], lt["area_m2"])
    wn.get_node(leak_node).emitter_coefficient = ec

    avg_p, avg_q, avg_d = run_sim(wn, demand_level)

    p_s1 = avg_p.get("J_S1", 0)
    p_s2 = avg_p.get("J_S2", 0)
    sp_str = make_sensor_str(p_s1, p_s2)
    ss_str = f"{SENSOR_POSITIONS[0]:.3f},{SENSOR_POSITIONS[1]:.3f}"

    # Recover the original pipe ID from the split-pipe naming
    orig_pipe_id = None
    pipe_len = 0
    pipe_diam = 0.1524
    pipe_rough = 135
    for pid2, p2 in wn.pipes():
        if pid2.endswith("_L"):
            orig_pipe_id = pid2[:-2]
            pipe_diam = p2.diameter
            pipe_rough = p2.roughness
            pipe_len += p2.length
        elif pid2.endswith("_R") and pid2[:-2] == orig_pipe_id:
            pipe_len += p2.length

    # Average flow through the main spine (skip reservoir pipes)
    flows = [avg_q.get(pid, 0) for pid, _ in wn.pipes()
             if pid not in RESERVOIR_PIPES and not pid.startswith("P_R")]
    Q = np.mean(flows) if flows else 0
    area_p = math.pi * (pipe_diam ** 2) / 4
    vel = Q / area_p if area_p > 0 else 0

    leak_pressure = avg_p.get(leak_node, 0)
    leak_flow = abs(avg_d.get(leak_node, 0))

    left_sens = [s for s in SENSOR_POSITIONS if s < pos]
    right_sens = [s for s in SENSOR_POSITIONS if s > pos]
    left1 = pos - left_sens[-1] if len(left_sens) >= 1 else 0
    left2 = pos - left_sens[-2] if len(left_sens) >= 2 else 0
    right1 = right_sens[0] - pos if len(right_sens) >= 1 else 0
    right2 = right_sens[1] - pos if len(right_sens) >= 2 else 0

    def sens_press(side_sens, idx):
        if len(side_sens) > idx:
            sn = SENSOR_NODES[SENSOR_POSITIONS.index(side_sens[idx])]
            return avg_p.get(sn, 0)
        return 0

    return {
        "Pipe_ID": orig_pipe_id or "",
        "Pipe_Length_m": pipe_len,
        "Pipe_Diameter_m": pipe_diam,
        "Pipe_Roughness": pipe_rough,
        "Pipe_Material": MATERIAL,
        "Sensor_Positions_m": ss_str,
        "All_Sensor_Pressures_m": sp_str,
        "Avg_Flow_Velocity_mps": vel,
        "Avg_Flow_Rate_lps": Q * 1000,
        "Acoustic_Propagation_Speed_mps": ACOUSTIC_SPEED,
        "Attenuation_Alpha_per_m": ATTENUATION_ALPHA,
        "Leak_Status": 1,
        "Number_of_Leaks": 1,
        "Leak_Type": leak_type_name,
        "Leak_Area_m2": lt["area_m2"],
        "Leak_Flow_Lps": leak_flow,
        "Avg_Pressure_at_Leak": leak_pressure,
        "True_Leak_Position_m": pos,
        "Leak_Distance_2nd_Left_m": left2,
        "Leak_Distance_Left_m": left1,
        "Leak_Distance_Right_m": right1,
        "Leak_Distance_2nd_Right_m": right2,
        "Pipe_Pressure_Left1_m": sens_press(left_sens, 0),
        "Pipe_Pressure_Left2_m": sens_press(left_sens, 1),
        "Pipe_Pressure_Right1_m": sens_press(right_sens, 0),
        "Pipe_Pressure_Right2_m": sens_press(right_sens, 1),
        "Demand_Level": demand_level,
    }


# ======================================================
# MAIN
# ======================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    import sys
    orig_print = sys.stdout
    print("=" * 60)
    orig_print("mend.inp → CSV Converter")
    orig_print("=" * 60)

    orig_print("\n[1/3] Creating leak position templates...")
    create_templates()

    lt_weights = [lt["weight"] for lt in LEAK_TYPES]
    lt_cumsum = np.cumsum(lt_weights)
    lt_labels = [lt["name"] for lt in LEAK_TYPES]

    template_map = {}
    for pos in LEAK_POSITIONS:
        pl = f"pos_{pos}".replace(".", "_")
        template_map[pos] = os.path.join(TEMPLATE_DIR, f"{pl}.inp")

    all_rows = []

    # --- No-leak ---
    orig_print(f"\n[2/3] Generating {TOTAL_NOLEAK_SCENARIOS} no-leak scenarios...")
    for dl_idx, demand in enumerate(DEMAND_LEVELS):
        count = TOTAL_NOLEAK_SCENARIOS // len(DEMAND_LEVELS)
        if dl_idx < TOTAL_NOLEAK_SCENARIOS % len(DEMAND_LEVELS):
            count += 1

        rows = gen_no_leak_rows(demand, count)
        all_rows.extend(rows)
        orig_print(f"  Demand {demand} LPS: {len(rows)} rows")

    orig_print(f"  Total no-leak: {len(all_rows)} rows")

    # --- Leak ---
    orig_print(f"\n[3/3] Generating {TOTAL_LEAK_SCENARIOS} leak scenarios...")
    n_pos = len(LEAK_POSITIONS)
    n_dem = len(DEMAND_LEVELS)
    per_pos = TOTAL_LEAK_SCENARIOS // n_pos

    scenario_id = 0
    for pos in LEAK_POSITIONS:
        tmpl = template_map[pos]
        sub_count = per_pos
        for dl_idx, demand in enumerate(DEMAND_LEVELS):
            cnt = sub_count // n_dem
            if dl_idx < sub_count % n_dem:
                cnt += 1
            if cnt == 0:
                continue

            for _ in tqdm(range(cnt), desc=f"  pos={pos}m d={demand}",
                          file=sys.stdout):
                r = random.random()
                lt_idx = np.searchsorted(lt_cumsum, r)
                lt_name = lt_labels[lt_idx]
                scenario_id += 1
                row = gen_leak_row(pos, demand, lt_name, scenario_id, tmpl)
                all_rows.append(row)
            orig_print(f"    Demand {demand}: {cnt} scenarios")

    orig_print(f"  Total leak: {scenario_id} scenarios")

    # --- Save ---
    orig_print(f"\nSaving {len(all_rows)} rows to CSV...")
    df = pd.DataFrame(all_rows)
    cols = [
        "Pipe_ID", "Pipe_Length_m", "Pipe_Diameter_m", "Pipe_Roughness",
        "Pipe_Material", "Sensor_Positions_m", "All_Sensor_Pressures_m",
        "Avg_Flow_Velocity_mps", "Avg_Flow_Rate_lps",
        "Acoustic_Propagation_Speed_mps", "Attenuation_Alpha_per_m",
        "Leak_Status", "Number_of_Leaks", "Leak_Type",
        "Leak_Area_m2", "Leak_Flow_Lps", "Avg_Pressure_at_Leak",
        "True_Leak_Position_m",
        "Leak_Distance_2nd_Left_m", "Leak_Distance_Left_m",
        "Leak_Distance_Right_m", "Leak_Distance_2nd_Right_m",
        "Pipe_Pressure_Left1_m", "Pipe_Pressure_Left2_m",
        "Pipe_Pressure_Right1_m", "Pipe_Pressure_Right2_m",
        "Demand_Level",
    ]
    df = df[cols]
    out = os.path.join(OUTPUT_DIR, "mend.csv")
    df.to_csv(out, index=False)
    orig_print(f"\nDone! → {out}")
    orig_print(f"  Rows: {len(df)}")
    orig_print(f"  Leak: {df['Leak_Status'].sum()}  No-leak: {(df['Leak_Status']==0).sum()}")


if __name__ == "__main__":
    main()
