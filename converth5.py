

import os
import torch
import h5py

def is_state_dict(obj):
    return isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values())

for name in os.listdir("processed"):
    for sample in os.listdir(f"processed/{name}"):
        model_path = f"processed/{name}/{sample}"
        
        if not model_path.endswith((".pt", ".pth")):
            continue

        model_data = torch.load(model_path, map_location='cpu')

        h5_name = os.path.splitext(sample)[0] + ".h5"
        h5_path = f"processed/{name}/{h5_name}"

        with h5py.File(h5_path, 'w') as h5f:
            if is_state_dict(model_data):
                for key, value in model_data.items():
                    h5f.create_dataset(key, data=value.numpy())
            elif isinstance(model_data, torch.Tensor):
                h5f.create_dataset("tensor", data=model_data.numpy())
            else:
                print(f"⚠️ Unexpected type in {model_path}: {type(model_data)} — skipping.")
                continue

        print(f"✅ Converted {model_path} → {h5_path}")


import shutil
shutil.make_archive("processed","zip","processed")