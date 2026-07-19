import wntr
import math

def calculate_sensors(inp_file):
    # Load EPANET model
    wn = wntr.network.WaterNetworkModel(inp_file)

    total_sensors = 0
    pipe_sensor_details = {}

    for pipe_name, pipe in wn.pipes():
        length = pipe.length  # length in meters (assuming SI units)

        if length <= 500:
            sensors = 1
        elif length <= 1500:
            sensors = 2
        else:
            sensors = math.floor(length / 500)

        pipe_sensor_details[pipe_name] = {
            "length_m": length,
            "sensors": sensors
        }

        total_sensors += sensors

    return total_sensors, pipe_sensor_details


# Example usage
inp_path = "../inp/NW_Model1.inp"
total, details = calculate_sensors(inp_path)

print("Total Sensors Required:", total)

for pipe, info in details.items():
    print(f"{pipe}: Length = {info['length_m']:.2f} m → Sensors = {info['sensors']}")