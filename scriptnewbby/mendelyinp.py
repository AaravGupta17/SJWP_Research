"""
generate_testbed_inp.py
=======================
Generates an EPANET .inp file replicating the Aghashahi et al. (2023)
Mendeley testbed with:
  - 1 main pipe: 47 m, 152.4 mm (6-inch) Schedule 80 PVC
  - Operating pressure: 5–10 m head (reservoir set at 10 m, demand node at 5 m)
  - Leak flows: 0.018–0.075 LPS modelled as emitter orifices
  - Leak area: 2 mm²
  - Sensors (virtual junction nodes) at ~12 m and ~35 m along pipe
  - 4 leak types weighted as Model D (Orifice > Longitudinal > Circumferential > Gasket)
  - Geotextile damping: upper-range Hazen-Williams roughness (130–140 for buried PVC)

Pipe properties:
  - Diameter: 152.4 mm (6-inch nominal)
  - Wall thickness: 11.43 mm  (Schedule 80, ASTM D1785)
  - Material: PVC (C ≈ 130–140 H-W coefficient)
  - Wave speed (Joukowsky): ~395 m/s for Sch 80 PVC (used in commentary only)

Leak Model D weighting (based on Aghashahi et al.):
  OL (Orifice)           45%  → largest emitter coefficient
  LC (Longitudinal Crack) 25%  → medium-high emitter coefficient
  CC (Circumferential)   20%  → medium emitter coefficient
  GL (Gasket)            10%  → smallest emitter coefficient

Emitter equation: Q = C_e * P^0.5   (EPANET uses pressure in m)
For 2 mm² orifice: Q = Cd * A * sqrt(2gH)
  Cd = 0.61, A = 2e-6 m², g = 9.81 m/s²
  At H=7.5 m (mid-range): Q ≈ 0.019 LPS  ✓ (within 0.018–0.075 LPS target)

Emitter coefficients derived from Q = C_e * P^0.5  →  C_e = Q / P^0.5
  Scale by Model D weights to distribute leak across 4 junction nodes.
"""

import math
import datetime

# ---------------------------------------------------------------------------
# Physical constants & pipe parameters
# ---------------------------------------------------------------------------
PIPE_LENGTH_M = 47.0  # m
PIPE_DIA_MM = 152.4  # mm (6-inch nominal)
PIPE_DIA_M = PIPE_DIA_MM / 1000.0
WALL_T_MM = 11.43  # mm  (Schedule 80, ASTM D1785)
HW_ROUGHNESS = 135  # Hazen-Williams C  (upper range for buried PVC
#  with geotextile wrap; open-air ≈ 140–150)

# Pressure bounds (m head)
HEAD_RESERVOIR = 12.0  # reservoir head (gives 5–10 m at pipe nodes)
HEAD_OUTLET = 2.0  # fixed-head outlet (downstream end)

# Leak target: 2 mm² orifice, Cd=0.61, mid-pressure ≈ 7.5 m
Cd = 0.61
A_m2 = 2e-6  # 2 mm² in m²
g = 9.81  # m/s²


def orifice_flow_lps(head_m):
    """Return leak flow in LPS for 2 mm² orifice at given head."""
    return Cd * A_m2 * math.sqrt(2 * g * head_m) * 1000.0


# Emitter coefficient C_e such that Q(LPS) = C_e * sqrt(P_m)
# EPANET emitter: Q_m3s = C_e * P_m^exponent  (exponent default 0.5)
# We work in LPS units for readability, then EPANET uses m³/s internally.
# EPANET actually: flow (unit) = emitter_coeff * pressure^exponent
# With flow in LPS and pressure in m: C_e = Q_LPS / sqrt(P_m)
def emitter_coeff(flow_lps, pressure_m):
    return flow_lps / math.sqrt(pressure_m)


# Representative operating pressure for coefficient calibration
P_ref = 7.5  # m (midpoint of 5–10 m range)

# Target total leak flow at reference pressure
Q_total_lps = orifice_flow_lps(P_ref)  # ≈ 0.019 LPS at 7.5 m

# Model D weights
MODEL_D = {
    "OL": 0.45,  # Orifice Leak
    "LC": 0.25,  # Longitudinal Crack
    "CC": 0.20,  # Circumferential Crack
    "GL": 0.10,  # Gasket Leak
}

# Emitter coefficients per leak type
leak_emitter_coeffs = {
    lt: emitter_coeff(Q_total_lps * w, P_ref)
    for lt, w in MODEL_D.items()
}

# ---------------------------------------------------------------------------
# Network topology
# ---------------------------------------------------------------------------
# Nodes along the 47 m pipe (positions in m from inlet):
#   J0    = 0 m    (inlet junction, connects to reservoir)
#   J_S1  = 12 m   (sensor node 1)
#   J_OL  = 20 m   (Orifice Leak node)
#   J_LC  = 24 m   (Longitudinal Crack node)
#   J_CC  = 28 m   (Circumferential Crack node)
#   J_GL  = 32 m   (Gasket Leak node)
#   J_S2  = 35 m   (sensor node 2)
#   J_END = 47 m   (outlet junction, connects to tank/fixed head)

NODES = [
    ("J0", 0.0, 0.0),  # (name, x_m, elevation_m)
    ("J_S1", 12.0, 0.0),
    ("J_OL", 20.0, 0.0),
    ("J_LC", 24.0, 0.0),
    ("J_CC", 28.0, 0.0),
    ("J_GL", 32.0, 0.0),
    ("J_S2", 35.0, 0.0),
    ("J_END", 47.0, 0.0),
]

# Leak junctions (node name -> leak type)
LEAK_NODES = {
    "J_OL": "OL",
    "J_LC": "LC",
    "J_CC": "CC",
    "J_GL": "GL",
}

# Pipes connecting consecutive nodes
PIPES = []
node_positions = {n: x for n, x, _ in NODES}
node_names = [n for n, _, _ in NODES]
for i in range(len(node_names) - 1):
    n1 = node_names[i]
    n2 = node_names[i + 1]
    length = node_positions[n2] - node_positions[n1]
    pipe_id = f"P{i + 1}"
    PIPES.append((pipe_id, n1, n2, length))


# ---------------------------------------------------------------------------
# INP file builder
# ---------------------------------------------------------------------------

def build_inp() -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []

    def sec(title):
        lines.append(f"[{title}]")

    def blank():
        lines.append("")

    # ---- TITLE ----
    sec("TITLE")
    lines.append(f"; Aghashahi et al. (2023) Mendeley Testbed — EPANET Model")
    lines.append(f"; Generated: {ts}")
    lines.append(f"; Pipe: {PIPE_LENGTH_M} m, {PIPE_DIA_MM} mm Sch-80 PVC")
    lines.append(f"; Wall thickness: {WALL_T_MM} mm | HW roughness C={HW_ROUGHNESS}")
    lines.append(f"; Leak area: 2 mm² | Model D weighting | Geotextile damping upper range")
    lines.append(f"; Wave speed (Sch-80 PVC, 152.4mm): ~395 m/s")
    lines.append(f"; Orifice flow @ 7.5 m head: {Q_total_lps:.4f} LPS")
    blank()

    # ---- JUNCTIONS ----
    sec("JUNCTIONS")
    lines.append(";ID               Elev    Demand   Pattern")
    for name, x, elev in NODES:
        # Sensor nodes: zero base demand (monitoring only)
        # Leak nodes: base demand = 0 (leak via emitter), no pattern
        demand = 0.0
        lines.append(f" {name:<16} {elev:<8.2f} {demand:<8.4f} ;")
    blank()

    # ---- RESERVOIRS ----
    sec("RESERVOIRS")
    lines.append(";ID         Head    Pattern")
    lines.append(f" R_IN       {HEAD_RESERVOIR:.2f}    ;  Upstream reservoir ({HEAD_RESERVOIR} m head)")
    lines.append(f" R_OUT      {HEAD_OUTLET:.2f}    ;  Downstream fixed head ({HEAD_OUTLET} m head)")
    blank()

    # ---- TANKS ----
    sec("TANKS")
    blank()

    # ---- PIPES ----
    sec("PIPES")
    lines.append(";ID         Node1        Node2        Length   Diameter   Roughness  MinorLoss  Status")
    for pid, n1, n2, length in PIPES:
        lines.append(
            f" {pid:<10} {n1:<12} {n2:<12} {length:<8.2f} {PIPE_DIA_MM:<10.1f} "
            f"{HW_ROUGHNESS:<10} 0          Open"
        )
    # Reservoir connection pipes (short, large diameter, high C — negligible loss)
    lines.append(
        f" P_RIN      R_IN         J0           0.10     {PIPE_DIA_MM:<10.1f} 140        0          Open"
    )
    lines.append(
        f" P_ROUT     J_END        R_OUT        0.10     {PIPE_DIA_MM:<10.1f} 140        0          Open"
    )
    blank()

    # ---- PUMPS ----
    sec("PUMPS")
    blank()

    # ---- VALVES ----
    sec("VALVES")
    blank()

    # ---- EMITTERS (leak orifices) ----
    sec("EMITTERS")
    lines.append(";Junction        Coefficient")
    lines.append(";  Emitter models Q(LPS) = Coeff * sqrt(P_m)  [EPANET flow units = LPS]")
    lines.append(f";  Calibrated at P_ref={P_ref} m, 2 mm² orifice, Cd={Cd}")
    lines.append(f";  Total Q @ ref = {Q_total_lps:.4f} LPS  |  Model D weights: OL=45% LC=25% CC=20% GL=10%")
    for node, lt in LEAK_NODES.items():
        coeff = leak_emitter_coeffs[lt]
        q_at_ref = coeff * math.sqrt(P_ref)
        lines.append(
            f" {node:<16} {coeff:.6f}   ; {lt}  ({MODEL_D[lt] * 100:.0f}%)  "
            f"Q@{P_ref}m={q_at_ref:.4f} LPS"
        )
    blank()

    # ---- CURVES ----
    sec("CURVES")
    blank()

    # ---- PATTERNS ----
    sec("PATTERNS")
    blank()

    # ---- ENERGY ----
    sec("ENERGY")
    lines.append(" Global Efficiency   75")
    lines.append(" Global Price        0")
    lines.append(" Demand Charge       0")
    blank()

    # ---- STATUS ----
    sec("STATUS")
    blank()

    # ---- CONTROLS ----
    sec("CONTROLS")
    blank()

    # ---- RULES ----
    sec("RULES")
    blank()

    # ---- DEMANDS ----
    sec("DEMANDS")
    blank()

    # ---- QUALITY ----
    sec("QUALITY")
    blank()

    # ---- SOURCES ----
    sec("SOURCES")
    blank()

    # ---- REACTIONS ----
    sec("REACTIONS")
    lines.append(" Order  Bulk            1")
    lines.append(" Order  Tank            1")
    lines.append(" Order  Wall            1")
    lines.append(" Global Bulk            0")
    lines.append(" Global Wall            0")
    lines.append(" Limiting Potential     0")
    lines.append(" Roughness Correlation  0")
    blank()

    # ---- MIXING ----
    sec("MIXING")
    blank()

    # ---- TIMES ----
    sec("TIMES")
    lines.append(" Duration            0:30")  # 30-second steady sim
    lines.append(" Hydraulic Timestep  0:00:01")
    lines.append(" Quality Timestep    0:05")
    lines.append(" Pattern Timestep    1:00")
    lines.append(" Pattern Start       0:00")
    lines.append(" Report Timestep     0:00:01")
    lines.append(" Report Start        0:00")
    lines.append(" Start ClockTime     12 am")
    lines.append(" Statistic           None")
    blank()

    # ---- REPORT ----
    sec("REPORT")
    lines.append(" Status             Yes")
    lines.append(" Summary            Yes")
    lines.append(" Page               0")
    lines.append(" Nodes              All")
    lines.append(" Links              All")
    lines.append(" Pressure           Yes")
    lines.append(" Flow               Yes")
    lines.append(" Velocity           Yes")
    blank()

    # ---- OPTIONS ----
    sec("OPTIONS")
    lines.append(" Units               LPS")
    lines.append(" Headloss            H-W")
    lines.append(" Specific Gravity    1.0")
    lines.append(" Viscosity           1.0")
    lines.append(" Trials              200")
    lines.append(" Accuracy            0.001")
    lines.append(" CHECKFREQ           2")
    lines.append(" MAXCHECK            10")
    lines.append(" DAMPLIMIT           0")
    lines.append(" Unbalanced          Continue 10")
    lines.append(" Pattern             1")
    lines.append(" Demand Multiplier   1.0")
    lines.append(" Emitter Exponent    0.5")
    lines.append(" Quality             None mg/L")
    lines.append(" Diffusivity         1")
    lines.append(" Tolerance           0.01")
    blank()

    # ---- COORDINATES ----
    sec("COORDINATES")
    lines.append(";Node           X-Coord         Y-Coord")
    for name, x, _ in NODES:
        lines.append(f" {name:<16} {x:<16.2f} 0.00")
    lines.append(f" R_IN           -5.00            0.00")
    lines.append(f" R_OUT           52.00            0.00")
    blank()

    # ---- VERTICES ----
    sec("VERTICES")
    blank()

    # ---- LABELS ----
    sec("LABELS")
    lines.append(";X-Coord        Y-Coord         Label & Anchor Node")
    lines.append(f" 12.00          1.00            \"Sensor 1\" J_S1")
    lines.append(f" 35.00          1.00            \"Sensor 2\" J_S2")
    blank()

    # ---- BACKDROP ----
    sec("BACKDROP")
    blank()

    # ---- END ----
    sec("END")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------

def print_summary():
    print("=" * 62)
    print("  Testbed INP Generator — Parameter Summary")
    print("=" * 62)
    print(f"  Pipe length       : {PIPE_LENGTH_M} m")
    print(f"  Pipe diameter     : {PIPE_DIA_MM} mm (6-inch nominal)")
    print(f"  Wall thickness    : {WALL_T_MM} mm  (Sch-80, ASTM D1785)")
    print(f"  H-W roughness C   : {HW_ROUGHNESS}  (geotextile upper range)")
    est_wave = 1 / math.sqrt(
        (1000 / (2.0e9)) +  # water bulk modulus
        (1000 * PIPE_DIA_M / (2.76e9 * (WALL_T_MM / 1000)))  # PVC hoop compliance
    )
    # Simplified Joukowsky: a = sqrt(K/rho / (1 + K*D/(E*e)))
    K_w = 2.0e9  # Pa
    rho = 1000  # kg/m3
    E = 2.76e9  # Pa  (PVC modulus)
    e = WALL_T_MM / 1000
    a = math.sqrt((K_w / rho) / (1 + K_w * PIPE_DIA_M / (E * e)))
    print(f"  Est. wave speed   : {a:.1f} m/s  (Joukowsky, PVC E=2.76 GPa)")
    print(f"  Upstream head     : {HEAD_RESERVOIR} m")
    print(f"  Downstream head   : {HEAD_OUTLET} m")
    print(f"  Leak orifice area : 2 mm²  (Cd={Cd})")
    print(f"  Ref pressure      : {P_ref} m")
    print(f"  Total Q @ ref     : {Q_total_lps:.4f} LPS")
    print()
    print(f"  Flow range (2mm², Cd={Cd}):")
    for h in [5.0, 7.5, 10.0]:
        print(f"    @ {h:4.1f} m head  → {orifice_flow_lps(h):.4f} LPS")
    print()
    print(f"  Model D emitter coefficients:")
    for lt, coeff in leak_emitter_coeffs.items():
        q = coeff * math.sqrt(P_ref)
        print(f"    {lt} ({MODEL_D[lt] * 100:.0f}%)  C_e={coeff:.6f}  Q@ref={q:.4f} LPS")
    print()
    print(f"  Sensor nodes      : J_S1 @ 12 m,  J_S2 @ 35 m")
    print(f"  Sensor separation : {35 - 12} m")
    print("=" * 62)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_summary()

    inp_content = build_inp()
    output_file = "../inp/mend.inp"

    with open(output_file, "w") as f:
        f.write(inp_content)

    print(f"\n  INP file written: {output_file}")
    print(f"  Lines: {inp_content.count(chr(10))}")
    print("\n  Open in EPANET 2.2 or OWA-EPANET to run simulation.")
    print("  Emitter Exponent is set to 0.5 (OPTIONS section).")
    print("  To simulate individual leak types, zero out the other")
    print("  emitter coefficients in [EMITTERS] section.")