import wntr
import numpy as np

# ===== SET YOUR INP FILE PATH HERE =====
inp_file_path = r"../inp/Net6.inp"
# =======================================


# Load EPANET model
wn = wntr.network.WaterNetworkModel(inp_file_path)

# Get all pipe names
pipe_names = wn.pipe_name_list

if len(pipe_names) == 0:
    raise ValueError("No pipes found in the network.")

# Store pipe lengths in dictionary {pipe_name: length}
pipe_lengths = {}
for p in pipe_names:
    pipe = wn.get_link(p)
    pipe_lengths[p] = pipe.length

# Convert to numpy array for statistics
lengths = np.array(list(pipe_lengths.values()))

# Basic statistics
total_pipes = len(lengths)
total_length = np.sum(lengths)
average_length = np.mean(lengths)

# Largest and smallest pipes
largest_pipe = max(pipe_lengths, key=pipe_lengths.get)
shortest_pipe = min(pipe_lengths, key=pipe_lengths.get)

largest_length = pipe_lengths[largest_pipe]
shortest_length = pipe_lengths[shortest_pipe]

# Print results
print("----- Pipe Length Statistics -----")
print(f"Total number of pipes : {total_pipes}")
print(f"Total pipe length     : {total_length:.3f}")
print(f"Average pipe length   : {average_length:.3f}")
print()
print("----- Extreme Pipes -----")
print(f"Largest pipe  : {largest_pipe}  | Length = {largest_length:.3f}")
print(f"Shortest pipe : {shortest_pipe} | Length = {shortest_length:.3f}")