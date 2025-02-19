# 

import os

for name in os.listdir("data/"):
    if os.path.isdir(os.path.join("data/", name)):
        open(f"vods/{name}", "w")
