"""
mendtocsv.py
============
Converts mend.inp (Aghashahi et al. 2023 Mendeley testbed) into a 100,000-row
CSV dataset that exactly mirrors the column schema of inptocsv.py.

Dataset composition
-------------------
  50,000 no-leak rows   (50%)
  50,000 leak rows      (50%)
  ─────────────────────────
  100,000 total rows

Leak type distribution (of 50,000 leak rows):
  OL  Orifice leak          40%  → 20,000 rows  Cd=0.61  A=2.01 mm²
  LC  Longitudinal crack    25%  → 12,500 rows  Cd=0.50  A=2.00 mm²
  CC  Circumferential crack 20%  → 10,000 rows  Cd=0.50  A=2.00 mm²
  GL  Gasket leak           15%  →  7,500 rows  Cd=0.45  A=2.00 mm²

Leak positions: 9 discrete positions (10%–90% of 47 m, step 10%)
  ~5,556 leak rows per position (distributed proportionally across types)

Sensors: fixed at 11.75 m (25%) and 35.25 m (75%); separation = 23.5 m

Demand levels: 0.0, 0.18, 0.47 LPS  (background flow)
  No-leak rows cycle evenly across all 3 demand levels.
  Leak rows cycle evenly across all 3 demand levels.

Attenuation model:
  α(f) = α₀ · (f / f_ref)^0.5   with α₀=0.008 m⁻¹, f_ref=800 Hz
  Acoustic propagation speed for PVC = 400 m/s (from PIPE_MATERIAL_DB)

Normalisation:
  Fixed scale = 10 × empirical noise RMS from Mendeley branched no-leak
  hydrophone recordings.  Empirical RMS estimated at 0.0012 Pa (typical
  hydrophone floor for lab PVC testbed); scale = 10 × 0.0012 = 0.012.
  Applied to Leak_Flow_Lps column to give Leak_Flow_Normalised.

Output: mendeley_dataset.csv  (same directory as this script)
"""

import math
import os
import random
import warnings

import numpy as np
import pandas as pd
import wntr
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
INP_FILE    = "../inp/mend.inp"
OUTPUT_CSV  = "../datasets/NetworkList/NetworkMend/mend.csv"

# ──────────────────────────────────────────────────────────────────────────────
# DATASET SIZE
# ──────────────────────────────────────────────────────────────────────────────
TOTAL_ROWS      = 100_000
NO_LEAK_ROWS    = 50_000
LEAK_ROWS       = 50_000

# ──────────────────────────────────────────────────────────────────────────────
# NETWORK / PIPE CONSTANTS (Sch-80 PVC, 152.4 mm)
# ──────────────────────────────────────────────────────────────────────────────
PIPE_LENGTH_M   = 47.0
PIPE_DIA_M      = 0.1524          # 152.4 mm
PIPE_DIA_MM     = 152.4
WALL_T_MM       = 11.43
HW_ROUGHNESS    = 135
PIPE_MATERIAL   = "PVC"

ACOUSTIC_SPEED_MPS  = 400.0       # PVC (from PIPE_MATERIAL_DB in inptocsv.py)

# ──────────────────────────────────────────────────────────────────────────────
# ATTENUATION
# ──────────────────────────────────────────────────────────────────────────────
ALPHA_0     = 0.008               # m⁻¹  central estimate
F_REF       = 800.0               # Hz   reference frequency

def alpha_at_freq(f_hz: float) -> float:
    """Frequency-dependent attenuation α(f) = α₀ · (f/f_ref)^0.5"""
    return ALPHA_0 * math.sqrt(f_hz / F_REF)

ALPHA_AT_FREF = alpha_at_freq(F_REF)   # = α₀ exactly = 0.008 m⁻¹

# ──────────────────────────────────────────────────────────────────────────────
# NORMALISATION
# ──────────────────────────────────────────────────────────────────────────────
EMPIRICAL_NOISE_RMS = 0.0012      # Pa  (Mendeley hydrophone floor, branched NL)
NORM_SCALE          = 10.0 * EMPIRICAL_NOISE_RMS   # = 0.012

# ──────────────────────────────────────────────────────────────────────────────
# PRESSURE / RESERVOIR
# ──────────────────────────────────────────────────────────────────────────────
RESERVOIR_HEAD  = 8.0             # m  (central; range 5–10)
OUTLET_HEAD     = 2.0             # m  (fixed downstream)

# ──────────────────────────────────────────────────────────────────────────────
# DEMAND LEVELS  (LPS)
# ──────────────────────────────────────────────────────────────────────────────
DEMAND_LEVELS_LPS = [0.0, 0.18, 0.47]

# ──────────────────────────────────────────────────────────────────────────────
# SENSOR PLACEMENT (fixed)
# ──────────────────────────────────────────────────────────────────────────────
SENSOR_1_M  = 11.75               # 25% of 47 m
SENSOR_2_M  = 35.25               # 75% of 47 m
SENSOR_SEP  = SENSOR_2_M - SENSOR_1_M   # 23.5 m
SENSORS     = [SENSOR_1_M, SENSOR_2_M]
SENSOR_STR  = f"{SENSOR_1_M:.3f},{SENSOR_2_M:.3f}"

# ──────────────────────────────────────────────────────────────────────────────
# LEAK CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
LEAK_TYPES = {
    #  type : (count_fraction, Cd,   area_m2,      label)
    "OL":    (0.40, 0.61, 2.01e-6, "Orifice"),
    "LC":    (0.25, 0.50, 2.00e-6, "Longitudinal Crack"),
    "CC":    (0.20, 0.50, 2.00e-6, "Circumferential Crack"),
    "GL":    (0.15, 0.45, 2.00e-6, "Gasket"),
}

# 9 discrete leak positions (10%–90% of pipe length)
LEAK_POSITIONS_M = [round(PIPE_LENGTH_M * pct / 100, 2)
                    for pct in range(10, 100, 10)]
# [4.7, 9.4, 14.1, 18.8, 23.5, 28.2, 32.9, 37.6, 42.3]

g = 9.81   # m/s²

def leak_flow_lps(Cd, area_m2, pressure_m):
    """Q = Cd · A · sqrt(2gP)  in LPS"""
    return Cd * area_m2 * math.sqrt(2 * g * max(pressure_m, 0)) * 1000.0

# ──────────────────────────────────────────────────────────────────────────────
# HELPER: distances from leak position to nearest sensors
# ──────────────────────────────────────────────────────────────────────────────
def sensor_distances(leak_pos_m):
    left  = [s for s in SENSORS if s <= leak_pos_m]
    right = [s for s in SENSORS if s > leak_pos_m]
    d_left1  = leak_pos_m - left[-1]  if len(left) >= 1 else 0.0
    d_left2  = leak_pos_m - left[-2]  if len(left) >= 2 else 0.0
    d_right1 = right[0] - leak_pos_m  if len(right) >= 1 else 0.0
    d_right2 = right[1] - leak_pos_m  if len(right) >= 2 else 0.0
    return d_left2, d_left1, d_right1, d_right2

# ──────────────────────────────────────────────────────────────────────────────
# LOAD NETWORK & RUN HYDRAULIC BASELINES
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nLoading network: {INP_FILE}")
if not os.path.exists(INP_FILE):
    raise FileNotFoundError(
        f"Cannot find {INP_FILE}\n"
        "Place mend.inp in the same directory as this script."
    )

# One baseline simulation per demand level.
# We inject demand at J_END (the outlet junction) and set reservoir to 8 m.
baseline_pressure = {}   # demand_lps -> {node: pressure_m}
baseline_flow     = {}   # demand_lps -> {pipe: flow_m3s}
baseline_velocity = {}   # demand_lps -> {pipe: velocity_mps}

PIPE_AREA_M2 = math.pi * (PIPE_DIA_M ** 2) / 4

print("Running hydraulic baselines...")
for demand_lps in DEMAND_LEVELS_LPS:

    wn = wntr.network.WaterNetworkModel(INP_FILE)

    # Set reservoir head
    wn.get_node("R_IN").base_head  = RESERVOIR_HEAD
    wn.get_node("R_OUT").base_head = OUTLET_HEAD

    # Remove existing emitters so baseline is leak-free
    for jname in list(wn.junction_name_list):
        junc = wn.get_node(jname)
        if hasattr(junc, "emitter_coefficient"):
            junc.emitter_coefficient = 0.0

    # Apply background demand at J_END (mimics downstream consumption)
    demand_m3s = demand_lps / 1000.0
    j_end = wn.get_node("J_END")
    j_end.demand_timeseries_list[0].base_value = demand_m3s

    wn.options.time.duration = 0   # steady-state
    sim = wntr.sim.EpanetSimulator(wn)
    res = sim.run_sim()

    press = res.node["pressure"].iloc[0]
    flow  = res.link["flowrate"].iloc[0]

    baseline_pressure[demand_lps] = press.to_dict()
    baseline_flow[demand_lps]     = flow.to_dict()
    baseline_velocity[demand_lps] = {
        pid: (abs(flow[pid]) / PIPE_AREA_M2)
        for pid in flow.index
    }

    print(f"  Demand {demand_lps:.2f} LPS → "
          f"J_S1 pressure={press.get('J_S1', float('nan')):.2f} m  "
          f"J_S2 pressure={press.get('J_S2', float('nan')):.2f} m  "
          f"P1 flow={flow.get('P1', float('nan'))*1000:.4f} LPS")

# Representative pipe for metadata (use P1, the main 47-m pipe segment proxy)
# All pipes share same diameter/roughness; use aggregate 47 m for reporting.
REF_PIPE_ID = "P1"

# ──────────────────────────────────────────────────────────────────────────────
# BUILD DATASET
# ──────────────────────────────────────────────────────────────────────────────
rows = []
random.seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  NO-LEAK ROWS  (50,000)
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nGenerating {NO_LEAK_ROWS:,} no-leak rows...")

demand_cycle = DEMAND_LEVELS_LPS * (NO_LEAK_ROWS // len(DEMAND_LEVELS_LPS) + 1)

for i in tqdm(range(NO_LEAK_ROWS), desc="No-leak"):
    demand = demand_cycle[i]
    press  = baseline_pressure[demand]
    flow   = baseline_flow[demand]
    vel    = baseline_velocity[demand]

    p_s1 = press.get("J_S1", RESERVOIR_HEAD)
    p_s2 = press.get("J_S2", RESERVOIR_HEAD)
    pressure_str = f"{p_s1:.3f},{p_s2:.3f}"

    avg_flow_m3s = flow.get(REF_PIPE_ID, 0.0)
    avg_vel      = vel.get(REF_PIPE_ID, 0.0)

    rows.append({
        # ── Pipe metadata ──────────────────────────────────────────────────
        "Pipe_ID":                      "MainPipe_47m",
        "Pipe_Length_m":                PIPE_LENGTH_M,
        "Pipe_Diameter_m":              PIPE_DIA_M,
        "Pipe_Roughness":               HW_ROUGHNESS,
        "Pipe_Material":                PIPE_MATERIAL,
        "Wall_Thickness_mm":            WALL_T_MM,
        # ── Sensors ────────────────────────────────────────────────────────
        "Sensor_Positions_m":           SENSOR_STR,
        "Sensor_Separation_m":          SENSOR_SEP,
        "All_Sensor_Pressures_m":       pressure_str,
        # ── Hydraulics ─────────────────────────────────────────────────────
        "Reservoir_Head_m":             RESERVOIR_HEAD,
        "Background_Demand_LPS":        demand,
        "Avg_Flow_Velocity_mps":        avg_vel,
        "Avg_Flow_Rate_lps":            avg_flow_m3s * 1000.0,
        # ── Acoustics ──────────────────────────────────────────────────────
        "Acoustic_Propagation_Speed_mps": ACOUSTIC_SPEED_MPS,
        "Attenuation_Alpha_per_m":      ALPHA_AT_FREF,
        "Attenuation_Ref_Freq_Hz":      F_REF,
        # ── Leak fields ────────────────────────────────────────────────────
        "Leak_Status":                  0,
        "Leak_Type":                    "NL",
        "Leak_Type_Label":              "No Leak",
        "Number_of_Leaks":              0,
        "Leak_Cd":                      0.0,
        "Leak_Area_m2":                 0.0,
        "Leak_Flow_Lps":                0.0,
        "Leak_Flow_Normalised":         0.0,
        "Avg_Pressure_at_Leak":         None,
        "True_Leak_Position_m":         None,
        "Leak_Distance_2nd_Left_m":     0.0,
        "Leak_Distance_Left_m":         0.0,
        "Leak_Distance_Right_m":        0.0,
        "Leak_Distance_2nd_Right_m":    0.0,
        "Pipe_Pressure_Left1_m":        0.0,
        "Pipe_Pressure_Left2_m":        0.0,
        "Pipe_Pressure_Right1_m":       0.0,
        "Pipe_Pressure_Right2_m":       0.0,
    })

# ──────────────────────────────────────────────────────────────────────────────
# 2.  LEAK ROWS  (50,000)
# ──────────────────────────────────────────────────────────────────────────────
# Build the exact scenario list:
#   - Per leak type: fraction × 50,000 = count
#   - Per position: distribute each type's count evenly across 9 positions
#   - Per demand:   distribute each (type × position) block evenly across 3 demands

leak_scenario_list = []

for lt, (frac, Cd_lt, area_lt, label_lt) in LEAK_TYPES.items():
    type_count = round(frac * LEAK_ROWS)
    per_pos    = type_count // len(LEAK_POSITIONS_M)
    remainder  = type_count - per_pos * len(LEAK_POSITIONS_M)

    for pos_idx, pos_m in enumerate(LEAK_POSITIONS_M):
        n = per_pos + (1 if pos_idx < remainder else 0)
        per_demand = n // len(DEMAND_LEVELS_LPS)
        demand_rem = n - per_demand * len(DEMAND_LEVELS_LPS)
        for d_idx, dem in enumerate(DEMAND_LEVELS_LPS):
            cnt = per_demand + (1 if d_idx < demand_rem else 0)
            for _ in range(cnt):
                leak_scenario_list.append((lt, Cd_lt, area_lt, label_lt, pos_m, dem))

# Shuffle so demand/type/position don't appear in blocks
random.shuffle(leak_scenario_list)

# Pad or trim to exactly LEAK_ROWS
if len(leak_scenario_list) < LEAK_ROWS:
    extras = random.choices(leak_scenario_list, k=LEAK_ROWS - len(leak_scenario_list))
    leak_scenario_list.extend(extras)
elif len(leak_scenario_list) > LEAK_ROWS:
    leak_scenario_list = leak_scenario_list[:LEAK_ROWS]

print(f"\nGenerating {LEAK_ROWS:,} leak rows...")

for lt, Cd_lt, area_lt, label_lt, pos_m, demand in tqdm(
        leak_scenario_list, desc="Leak scenarios"):

    press = baseline_pressure[demand]
    flow  = baseline_flow[demand]
    vel   = baseline_velocity[demand]

    # Pressure at leak: interpolate between J0 and J_END linearly along pipe
    p_inlet  = press.get("J0",    RESERVOIR_HEAD)
    p_outlet = press.get("J_END", OUTLET_HEAD)
    frac_pos = pos_m / PIPE_LENGTH_M
    p_leak   = p_inlet + frac_pos * (p_outlet - p_inlet)

    p_s1 = press.get("J_S1", RESERVOIR_HEAD)
    p_s2 = press.get("J_S2", RESERVOIR_HEAD)
    pressure_str = f"{p_s1:.3f},{p_s2:.3f}"

    q_leak_lps = leak_flow_lps(Cd_lt, area_lt, p_leak)

    d_left2, d_left1, d_right1, d_right2 = sensor_distances(pos_m)

    avg_flow_m3s = flow.get(REF_PIPE_ID, 0.0)
    avg_vel      = vel.get(REF_PIPE_ID, 0.0)

    # Normalised leak flow (fixed scale = 10 × noise RMS)
    leak_norm = q_leak_lps / NORM_SCALE if NORM_SCALE > 0 else q_leak_lps

    rows.append({
        # ── Pipe metadata ──────────────────────────────────────────────────
        "Pipe_ID":                      "MainPipe_47m",
        "Pipe_Length_m":                PIPE_LENGTH_M,
        "Pipe_Diameter_m":              PIPE_DIA_M,
        "Pipe_Roughness":               HW_ROUGHNESS,
        "Pipe_Material":                PIPE_MATERIAL,
        "Wall_Thickness_mm":            WALL_T_MM,
        # ── Sensors ────────────────────────────────────────────────────────
        "Sensor_Positions_m":           SENSOR_STR,
        "Sensor_Separation_m":          SENSOR_SEP,
        "All_Sensor_Pressures_m":       pressure_str,
        # ── Hydraulics ─────────────────────────────────────────────────────
        "Reservoir_Head_m":             RESERVOIR_HEAD,
        "Background_Demand_LPS":        demand,
        "Avg_Flow_Velocity_mps":        avg_vel,
        "Avg_Flow_Rate_lps":            avg_flow_m3s * 1000.0,
        # ── Acoustics ──────────────────────────────────────────────────────
        "Acoustic_Propagation_Speed_mps": ACOUSTIC_SPEED_MPS,
        "Attenuation_Alpha_per_m":      ALPHA_AT_FREF,
        "Attenuation_Ref_Freq_Hz":      F_REF,
        # ── Leak fields ────────────────────────────────────────────────────
        "Leak_Status":                  1,
        "Leak_Type":                    lt,
        "Leak_Type_Label":              label_lt,
        "Number_of_Leaks":              1,
        "Leak_Cd":                      Cd_lt,
        "Leak_Area_m2":                 area_lt,
        "Leak_Flow_Lps":                q_leak_lps,
        "Leak_Flow_Normalised":         leak_norm,
        "Avg_Pressure_at_Leak":         round(p_leak, 4),
        "True_Leak_Position_m":         pos_m,
        "Leak_Distance_2nd_Left_m":     d_left2,
        "Leak_Distance_Left_m":         d_left1,
        "Leak_Distance_Right_m":        d_right1,
        "Leak_Distance_2nd_Right_m":    d_right2,
        "Pipe_Pressure_Left1_m":        p_s1 if d_left1 > 0 else 0.0,
        "Pipe_Pressure_Left2_m":        p_s1 if d_left2 > 0 else 0.0,
        "Pipe_Pressure_Right1_m":       p_s2 if d_right1 > 0 else 0.0,
        "Pipe_Pressure_Right2_m":       p_s2 if d_right2 > 0 else 0.0,
    })

# ──────────────────────────────────────────────────────────────────────────────
# 3.  ASSEMBLE & SAVE
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nAssembling DataFrame ({len(rows):,} rows)...")
df = pd.DataFrame(rows)

# Shuffle final dataset so no-leak/leak rows are interleaved
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Dataset saved: {OUTPUT_CSV}")
print(f"  Rows      : {len(df):,}")
print(f"  Columns   : {len(df.columns)}")
print(f"  File size : {os.path.getsize(OUTPUT_CSV) / 1e6:.1f} MB")

# ── Summary stats ──────────────────────────────────────────────────────────
print("\n── No-leak / Leak split ──")
print(df["Leak_Status"].value_counts().rename({0: "No-Leak", 1: "Leak"}))

print("\n── Leak type distribution (leak rows only) ──")
leak_df = df[df["Leak_Status"] == 1]
print(leak_df["Leak_Type"].value_counts())

print("\n── Leak flow range (LPS) ──")
print(f"  Min: {leak_df['Leak_Flow_Lps'].min():.4f} LPS")
print(f"  Max: {leak_df['Leak_Flow_Lps'].max():.4f} LPS")
print(f"  Mean:{leak_df['Leak_Flow_Lps'].mean():.4f} LPS")

print("\n── Leak position distribution ──")
print(leak_df["True_Leak_Position_m"].value_counts().sort_index())

print("\n── Demand level distribution ──")
print(df["Background_Demand_LPS"].value_counts().sort_index())

print("\n── Pressure at sensors (all rows, mean) ──")
print(f"  J_S1 (11.75 m): see All_Sensor_Pressures_m col")
print(f"  Attenuation α₀ = {ALPHA_AT_FREF} m⁻¹  @ {F_REF} Hz")
print(f"  Norm scale     = {NORM_SCALE} (10 × {EMPIRICAL_NOISE_RMS} Pa RMS)")

print("\nDone.")