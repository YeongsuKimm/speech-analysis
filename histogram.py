import os
import numpy as np
import matplotlib.pyplot as plt

# Path to dataset
data_path = "data/"

# Collect follower counts
follower_counts = []

# Loop through each streamer's folder
for streamer in os.listdir(data_path):
    streamer_path = os.path.join(data_path, streamer)
    metadata_path = os.path.join(streamer_path, f"{streamer}.txt")  # Metadata file

    if os.path.isfile(metadata_path):
        with open(metadata_path, "r") as f:
            for line in f:
                if "Followers:" in line:  # Adjust based on actual metadata format
                    count = np.log10(int(line.split(":")[1].strip()))
                    follower_counts.append(count)

# Plot histogram with logarithmic x-axis
plt.figure(figsize=(10, 6))
plt.hist(follower_counts, bins=3, edgecolor="black", log=True)  # log=True for log-scale y-axis
plt.xlabel("Follower Count (log scale)")
plt.ylabel("Frequency")
plt.title("Distribution of Followers Among Streamers (Log Scale)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Save the figure if running in a non-interactive environment
plt.savefig("histogram.png")
