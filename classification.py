import pandas as pd
import os


def get_dict(streamer_list):
    data = []
    for streamer in streamer_list:
        element = {}
        element["streamer"] = streamer
        with open(f"data/{streamer}/{streamer}.txt", "r+") as f:
            for line in f:
                if "Total Followers:" in line or "Followers:" in line:
                    raw_value = line.split(":")[1].strip()
                    try:
                        count = int(raw_value)
                        element["followers"] = count
                    except ValueError:
                        print(f"Skipping invalid follower count: {raw_value}")
        data.append(element)

    df = pd.DataFrame(data)
    df['follower_bin'] = pd.qcut(df['followers'], q=3, labels=[0, 1, 2])

    label_dict = dict(zip(df['streamer'], df['follower_bin']))
    return label_dict
# print(label_dict)


if __name__ == "__main__":
    data = []
    for streamer in os.listdir("data"):
        element = {}
        element["streamer"] = streamer
        with open(f"data/{streamer}/{streamer}.txt", "r+") as f:
            for line in f:
                if "Total Followers:" in line:
                    raw_value = line.split(":")[1].strip()
                    try:
                        count = int(raw_value)
                        element["followers"] = count
                    except ValueError:
                        print(f"Skipping invalid follower count: {raw_value}")
        data.append(element)

    df = pd.DataFrame(data)

    # Create follower bins using qcut
    # After creating follower_bins using qcut
    follower_bins = pd.qcut(df['followers'], q=3)

    # Get the bin intervals correctly
    bin_edges = follower_bins.cat.categories

    # Print the range of followers for each bin
    for i, bin_edge in enumerate(bin_edges):
        print(f"Bin {i} range: {bin_edge}")
    # If you want to access the specific left and right values
    # If you want to access the specific left and right values
    for i, bin_edge in enumerate(bin_edges):
        print(f"Bin {i} range: {bin_edge.left} to {bin_edge.right}")