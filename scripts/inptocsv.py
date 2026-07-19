import wntr
import numpy as np
import pandas as pd
import random
import os
from tqdm import tqdm
import math
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# ======================================================
# USER INPUT
# ======================================================
Cd = float(0.75)

# ======================================================
# NETWORK CONFIGURATION DICTIONARY
# ======================================================
NETWORKS = {
    "NW_Model1": {
        "inp": "../inp/NW_Model1.inp",
        "out": "../datasets/NetworkList/Network_1",
    },
    "Net3": {
        "inp": "../inp/Net3_(BWSN-2)_Morph_Error_Free_1s-WQ.inp",
        "out": "../datasets/NetworkList/Network_2",
    },
    "L-TOWN": {
        "inp": "../inp/L-TOWN.inp",
        "out": "../datasets/NetworkList/Network_3",
    },
    "BWSN": {
        "inp": "../inp/BWSN_Network.inp",
        "out": "../datasets/NetworkList/Network_4",
    },
    "KY9": {
        "inp": "../inp/ky9.inp",
        "out": "../datasets/NetworkList/Network_5",
    },
    "KY15": {
        "inp": "../inp/ky15.inp",
        "out": "../datasets/NetworkList/Network_6",
    },
    "Net6": {
        "inp": "../inp/Net6.inp",
        "out": "../datasets/NetworkList/Network_7",
    },
    "Rich": {
        "inp": "../inp/Richmond_skeleton.inp",
        "out": "../datasets/NetworkList/Network_8",
    }
}

TOTAL_SCENARIOS = 1200
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

# ======================================================
# SENSOR RULES (NEW POLICY)
# ======================================================
def get_sensor_positions(pipe_length):

    if pipe_length <= 750:
        # 2 sensors, trisect
        return [pipe_length / 3, 2 * pipe_length / 3]

    elif pipe_length <= 1200:
        # 3 equally spaced
        return [pipe_length / 4,
                pipe_length / 2,
                3 * pipe_length / 4]

    else:
        # 4 equally spaced
        return [pipe_length / 5,
                2 * pipe_length / 5,
                3 * pipe_length / 5,
                4 * pipe_length / 5]

# ======================================================
# MAIN LOOP OVER NETWORKS
# ======================================================
for network_name, cfg in NETWORKS.items():

    INP_FILE = cfg["inp"]
    BASE_OUTPUT_DIR = cfg["out"]

    print(f"\n==============================")
    print(f"Processing Network: {network_name}")
    print(f"==============================")

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    for material in MATERIALS:

        print(f"\n Material: {material}")
        out_dir = os.path.join(BASE_OUTPUT_DIR, material)
        os.makedirs(out_dir, exist_ok=True)

        pipes = list(wntr.network.WaterNetworkModel(INP_FILE).pipes())

        baseline_pressure = {}
        baseline_flow = {}

        # ======================================================
        # HYDRAULIC BASELINE (3 SIMS PER MATERIAL PER NETWORK)
        # ======================================================
        for demand in DEMAND_LEVELS:

            wn = wntr.network.WaterNetworkModel(INP_FILE)
            wn.options.time.duration = SIM_HOURS * 3600
            wn.options.hydraulic.demand_multiplier = demand

            sim = wntr.sim.EpanetSimulator(wn)
            res = sim.run_sim()

            baseline_pressure[demand] = res.node["pressure"].mean()
            baseline_flow[demand] = res.link["flowrate"].mean()

            # ================== BASE DATASET ==================
            base_rows = []

            for pid, pipe in wn.pipes():

                sensors = sorted(get_sensor_positions(pipe.length))
                sensor_str = ",".join([f"{s:.3f}" for s in sensors])

                base_pressure_val = baseline_pressure[demand][pipe.start_node_name]
                pressure_str = ",".join([f"{base_pressure_val:.3f}" for _ in sensors])

                Q = baseline_flow[demand][pid]
                area = math.pi * (pipe.diameter ** 2) / 4
                velocity = Q / area if area > 0 else 0

                base_rows.append({
                    "Pipe_ID": pid,
                    "Pipe_Length_m": pipe.length,
                    "Pipe_Diameter_m": pipe.diameter,
                    "Pipe_Roughness": pipe.roughness,
                    "Pipe_Material": material,
                    "Sensor_Positions_m": sensor_str,
                    "All_Sensor_Pressures_m": pressure_str,
                    "Avg_Flow_Velocity_mps": velocity,
                    "Avg_Flow_Rate_lps": Q * 1000,
                    "Acoustic_Propagation_Speed_mps": PIPE_MATERIAL_DB[material]["acoustic_speed"],
                    "Attenuation_Alpha_per_m": PIPE_MATERIAL_DB[material]["attenuation_alpha"],
                    "Leak_Status": 0,
                    "Number_of_Leaks": 0,
                    "Leak_Area_m2": 0.0,
                    "Leak_Flow_Lps": 0.0,
                    "Avg_Pressure_at_Leak": None,
                    "True_Leak_Position_m": None,
                    "Leak_Distance_2nd_Left_m": 0,
                    "Leak_Distance_Left_m": 0,
                    "Leak_Distance_Right_m": 0,
                    "Leak_Distance_2nd_Right_m": 0,
                    "Pipe_Pressure_Left1_m": 0,
                    "Pipe_Pressure_Left2_m": 0,
                    "Pipe_Pressure_Right1_m": 0,
                    "Pipe_Pressure_Right2_m": 0
                })

            pd.DataFrame(base_rows).to_csv(
                os.path.join(out_dir, f"base-{demand}.csv"),
                index=False
            )

        # ======================================================
        # FAST LEAK SCENARIOS
        # ======================================================
        for scenario_id in tqdm(range(1, TOTAL_SCENARIOS + 1),
                                desc=f"{network_name}-{material}"):

            leak_count = random.randint(1, max(1, int(0.3 * len(pipes))))
            selected = random.sample(pipes, leak_count)

            for demand in DEMAND_LEVELS:

                pressure_series = baseline_pressure[demand]
                flow_series = baseline_flow[demand]

                rows = []

                for pid, pipe in selected:

                    L = pipe.length
                    leak_position = random.uniform(0.05, 0.95) * L
                    leak_area = random.uniform(LEAK_AREA_MIN, LEAK_AREA_MAX)

                    sensors = sorted(get_sensor_positions(L))

                    left = [s for s in sensors if s < leak_position]
                    right = [s for s in sensors if s > leak_position]

                    left1 = leak_position - left[-1] if len(left) >= 1 else 0
                    left2 = leak_position - left[-2] if len(left) >= 2 else 0
                    right1 = right[0] - leak_position if len(right) >= 1 else 0
                    right2 = right[1] - leak_position if len(right) >= 2 else 0

                    avg_pressure = pressure_series[pipe.start_node_name]
                    leak_flow = Cd * leak_area * np.sqrt(2 * 9.81 * avg_pressure) * 1000

                    Q = flow_series[pid]
                    area = math.pi * (pipe.diameter ** 2) / 4
                    velocity = Q / area if area > 0 else 0

                    rows.append({
                        "Pipe_ID": pid,
                        "Pipe_Length_m": L,
                        "Pipe_Diameter_m": pipe.diameter,
                        "Pipe_Roughness": pipe.roughness,
                        "Pipe_Material": material,
                        "Sensor_Positions_m": ",".join([f"{s:.3f}" for s in sensors]),
                        "Avg_Flow_Velocity_mps": velocity,
                        "Avg_Flow_Rate_lps": Q * 1000,
                        "Acoustic_Propagation_Speed_mps": PIPE_MATERIAL_DB[material]["acoustic_speed"],
                        "Attenuation_Alpha_per_m": PIPE_MATERIAL_DB[material]["attenuation_alpha"],
                        "Leak_Status": 1,
                        "Number_of_Leaks": 1,
                        "Leak_Area_m2": leak_area,
                        "Leak_Flow_Lps": leak_flow,
                        "Avg_Pressure_at_Leak": avg_pressure,
                        "True_Leak_Position_m": leak_position,
                        "Leak_Distance_2nd_Left_m": left2,
                        "Leak_Distance_Left_m": left1,
                        "Leak_Distance_Right_m": right1,
                        "Leak_Distance_2nd_Right_m": right2,
                        "Pipe_Pressure_Left1_m": avg_pressure if left1 else 0,
                        "Pipe_Pressure_Left2_m": avg_pressure if left2 else 0,
                        "Pipe_Pressure_Right1_m": avg_pressure if right1 else 0,
                        "Pipe_Pressure_Right2_m": avg_pressure if right2 else 0
                    })

                filename = f"leak-{scenario_id}-{demand}.csv"
                pd.DataFrame(rows).to_csv(
                    os.path.join(out_dir, filename),
                    index=False
                )

print("\nALL NETWORKS PROCESSED SUCCESSFULLY")