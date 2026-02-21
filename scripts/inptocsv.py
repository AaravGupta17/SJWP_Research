import wntr
import numpy as np
import pandas as pd
import random
import os
from tqdm import tqdm

# ======================================================
# USER INPUT
# ======================================================
Cd = float(0.75)

# ======================================================
# CONFIG
# ======================================================
INP_FILE = "../NW_Model1.inp"
BASE_OUTPUT_DIR = "../test"
TOTAL_SCENARIOS = 1000
SIM_HOURS = 24

LEAK_AREA_MIN = 1e-6
LEAK_AREA_MAX = 2e-5

MATERIALS = ["DI", "CI", "PVC", "STEEL"]
DEMAND_LEVELS = [0.7, 1.0, 1.2]

PIPE_MATERIAL_DB = {
    "CI":    {"acoustic_speed": 1200.0, "attenuation_alpha": 0.0010},
    "DI":    {"acoustic_speed": 1000.0, "attenuation_alpha": 0.0005},
    "STEEL": {"acoustic_speed": 1300.0, "attenuation_alpha": 0.0003},
    "PVC":   {"acoustic_speed": 400.0,  "attenuation_alpha": 0.0030},
}

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# ======================================================
# SENSOR POSITIONS
# ======================================================
def get_sensor_positions(pipe_length):
    if pipe_length <= 500:
        return [0.5 * pipe_length]
    elif pipe_length <= 1500:
        return [pipe_length/3, 2*pipe_length/3]
    else:
        return list(np.arange(500, pipe_length, 500))

# ======================================================
# MAIN
# ======================================================
for material in MATERIALS:

    print(f"\n FAST Processing material: {material}")
    out_dir = os.path.join(BASE_OUTPUT_DIR, material)
    os.makedirs(out_dir, exist_ok=True)

    wn = wntr.network.WaterNetworkModel(INP_FILE)
    wn.options.time.duration = SIM_HOURS * 3600
    pipes = list(wn.pipes())

    # ======================================================
    # PRECOMPUTE BASELINE PRESSURES (ONLY 3 SIMULATIONS)
    # ======================================================
    baseline_pressures = {}

    for demand in DEMAND_LEVELS:

        wn_tmp = wntr.network.WaterNetworkModel(INP_FILE)
        wn_tmp.options.time.duration = SIM_HOURS * 3600
        wn_tmp.options.hydraulic.demand_multiplier = demand

        sim = wntr.sim.EpanetSimulator(wn_tmp)
        res = sim.run_sim()

        baseline_pressures[demand] = res.node["pressure"].mean()

        # Base dataset
        base_rows = []
        for pid, pipe in wn_tmp.pipes():
            base_rows.append({
                "Pipe_ID": pid,
                "Pipe_Length_m": pipe.length,
                "Pipe_Diameter_m": pipe.diameter,
                "Pipe_Roughness": pipe.roughness,
                "Pipe_Material": material,
                "Acoustic_Propagation_Speed_mps": PIPE_MATERIAL_DB[material]["acoustic_speed"],
                "Attenuation_Alpha_per_m": PIPE_MATERIAL_DB[material]["attenuation_alpha"],
                "Leak_Status": 0,
                "True_Leak_Position_m": None,
                "Leak_Distance_2nd_Left_m": 0,
                "Leak_Distance_Left_m": 0,
                "Leak_Distance_Right_m": 0,
                "Leak_Distance_2nd_Right_m": 0,
                "Number_of_Leaks": 0,
                "Leak_Area_m2": 0.0,
                "Leak_Flow_Lps": 0.0,
                "Avg_Pressure_at_Leak": None
            })

        pd.DataFrame(base_rows).to_csv(
            os.path.join(out_dir, f"base-{demand}.csv"), index=False
        )

    # ======================================================
    # FAST SCENARIO GENERATION
    # ======================================================
    for scenario_id in tqdm(range(1, TOTAL_SCENARIOS + 1),
                            desc=f"{material} Scenarios"):

        leak_count = random.randint(1, max(1, int(0.3 * len(pipes))))
        selected = random.sample(pipes, leak_count)

        for demand in DEMAND_LEVELS:

            pressure_series = baseline_pressures[demand]
            rows = []

            for pid, pipe in selected:
                L = pipe.length
                leak_position = random.uniform(0.05, 0.95) * L
                leak_area = random.uniform(LEAK_AREA_MIN, LEAK_AREA_MAX)

                avg_pressure = pressure_series[pipe.start_node_name]
                leak_flow = Cd * leak_area * np.sqrt(2 * 9.81 * avg_pressure) * 1000

                sensors = sorted(get_sensor_positions(L))

                left_sensors = [s for s in sensors if s < leak_position]
                right_sensors = [s for s in sensors if s > leak_position]

                left_dist = leak_position - left_sensors[-1] if len(left_sensors) >= 1 else 0
                left2_dist = leak_position - left_sensors[-2] if len(left_sensors) >= 2 else 0
                right_dist = right_sensors[0] - leak_position if len(right_sensors) >= 1 else 0
                right2_dist = right_sensors[1] - leak_position if len(right_sensors) >= 2 else 0

                rows.append({
                    "Pipe_ID": pid,
                    "Pipe_Length_m": L,
                    "Pipe_Diameter_m": pipe.diameter,
                    "Pipe_Roughness": pipe.roughness,
                    "Pipe_Material": material,
                    "Acoustic_Propagation_Speed_mps": PIPE_MATERIAL_DB[material]["acoustic_speed"],
                    "Attenuation_Alpha_per_m": PIPE_MATERIAL_DB[material]["attenuation_alpha"],
                    "Leak_Status": 1,
                    "True_Leak_Position_m": leak_position,
                    "Leak_Distance_2nd_Left_m": left2_dist,
                    "Leak_Distance_Left_m": left_dist,
                    "Leak_Distance_Right_m": right_dist,
                    "Leak_Distance_2nd_Right_m": right2_dist,
                    "Number_of_Leaks": 1,
                    "Leak_Area_m2": leak_area,
                    "Leak_Flow_Lps": leak_flow,
                    "Avg_Pressure_at_Leak": avg_pressure
                })

            filename = f"leak-{scenario_id}-{demand}.csv"
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, filename), index=False)

    print(f" Generated {TOTAL_SCENARIOS} FAST scenarios for {material}")

print("\n CLEAN DATASET GENERATION COMPLETE")