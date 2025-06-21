import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path to dataset
data_path = "data/"

# Collect follower counts
follower_counts = []
streamers = []
for streamer in os.listdir(data_path):
    streamer_path = os.path.join(data_path, streamer)
    metadata_path = os.path.join(streamer_path, f"{streamer}.txt")  # Metadata file

    if os.path.isfile(metadata_path):
        with open(metadata_path, "r") as f:
            for line in f:
                if "Current Subscribers:" in line:
                    raw_value = line.split(":")[1].strip()
                    try:
                        count = int(raw_value)
                        follower_counts.append(count)
                        if count ==2:
                            print(streamer)
                        streamers.append(streamer)
                    except ValueError:
                        print(f"Skipping invalid follower count: {raw_value}")

follower_counts= np.array(follower_counts)

print(len(streamers))
alls = []
for streamer in os.listdir("data"):
    alls.append(streamer)

for streamer in alls:
    if streamer not in streamers:
        print(streamer)

print(len(follower_counts))

print(max(follower_counts))
print(min(follower_counts))
# counts, bin_edges = np.histogram(follower_counts, bins=3)

# # Get the min and max of each bin
# bin_min_max = [(bin_edges[i], bin_edges[i+1]) for i in range(len(bin_edges)-1)]

# print(bin_min_max)

# Q1 = np.percentile(follower_counts, 25)
# Q3 = np.percentile(follower_counts, 75)

# IQR = Q3 - Q1

# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR

# filtered_data = follower_counts[(follower_counts >= lower_bound) & (follower_counts <= upper_bound)]

# # Plot histogram with logarithmic x-axis
# df = pd.DataFrame({"values":filtered_data})
# df['binned'] = pd.qcut(df['values'], q=3, labels=['Low', 'Medium', 'High'])

log_follower_counts = np.log10(follower_counts)

plt.figure(figsize=(10, 6))
bins = np.histogram(log_follower_counts, bins=4)[1]
plt.hist(follower_counts, bins=4, edgecolor="black", log=True)  # log=True for log-scale y-axis
plt.xlabel("Peak Sub Count")
plt.ylabel("Frequency")
plt.title("Distribution of Peak Sub Count Among Streamers")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Save the figure if running in a non-interactive environment
plt.savefig("scar.png")