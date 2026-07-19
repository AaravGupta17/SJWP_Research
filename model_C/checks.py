import pandas as pd
import numpy as np
from pathlib import Path

networks = {
    "Network_3": "L-TOWN",
    "Network_6": "KY15",
    "Network_8": "Richmond",
}

dataset_root = Path("../datasets/NetworkList")

for net_folder, net_name in networks.items():
    base_files = list((dataset_root / net_folder).rglob("base-1.0.csv"))
    df = pd.read_csv(base_files[0])
    df.columns = df.columns.str.strip()


    def effective_sep(val):
        try:
            parts = [float(x.strip()) for x in str(val).split(",") if x.strip()]
            if len(parts) >= 2:
                return parts[1] - parts[0]  # sensor[1] - sensor[0], what the model actually uses
            return np.nan
        except:
            return np.nan


    seps = df['Sensor_Positions_m'].apply(effective_sep).dropna()
    print(f"{net_name}: effective sep mean={seps.mean():.2f}m, median={seps.median():.2f}m")