import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path to dataset
data_path = "data/"

# print(len(os.listdir(data_path)))
# Collect follower counts
follower_counts = []
streamers = []
for streamer in os.listdir(data_path):
    streamer_path = os.path.join(data_path, streamer)
    metadata_path = os.path.join(streamer_path, f"{streamer}.txt")  # Metadata file

    if os.path.isfile(metadata_path):
        with open(metadata_path, "r") as f:
            for line in f:
                if "Total Followers:" in line or "Followers:" in line:
                    raw_value = line.split(":")[1].strip()
                    try:
                        count = int(raw_value)
                        follower_counts.append(count)
                        streamers.append(streamer)
                    except ValueError:
                        print(f"Skipping invalid follower count: {raw_value}")

follower_counts= np.array(follower_counts)

# print(follower_counts)

# print(len(streamers))

log_follower_counts = np.log10(follower_counts)

# print(log_follower_counts)

median = np.median(log_follower_counts)
q25 = np.percentile(log_follower_counts, 25)
q75 = np.percentile(log_follower_counts, 75)

# Deviations from the median
lower_dev = median - q25
upper_dev = q75 - median

print("Median:", median)
print("25th percentile:", q25)
print("75th percentile:", q75)
print("Deviation below median:", lower_dev)
print("Deviation above median:", upper_dev)

# print(max(log_follower_counts))
# print(min(log_follower_counts))

# print(max(follower_counts))
# print(min(follower_counts))

# plt.figure(figsize=(10, 6))
# plt.hist(log_follower_counts, bins=10, edgecolor="black", log=False)  # log=True for log-scale y-axis
# plt.xlabel("Log Follower Count", fontsize = 20)
# plt.ylabel("Frequency", fontsize = 20)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# # plt.title("Distribution of Follower Count Among Streamers")
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# # Save the figure if running in a non-interactive environment
# plt.savefig("histogram.png")