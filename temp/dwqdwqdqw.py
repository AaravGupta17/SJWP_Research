import time
import sys
from tqdm import tqdm
from colorama import init

# Initialize color handling (important for Windows / PyCharm)
init(autoreset=True)

print("Starting tqdm test...\n")

for i in tqdm(range(100),
              desc="Processing",
              colour="green",
              file=sys.stdout):
    time.sleep(0.05)

print("\nDone.")