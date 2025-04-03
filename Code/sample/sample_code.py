import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.dove_dataset import CreateDOVeDatasets
from torch.utils.data import DataLoader	

def get_dataloader_info(lodaer_data):
    print("Number of batches: ", len(lodaer_data))
    print("Total number of samples: ", len(lodaer_data.dataset))
    print(lodaer_data.dataset)
    print(vars(lodaer_data.dataset))

def visualize_frame(loader_data):
    # make sure the batch size is 1
    for clip in loader_data:
        clip = clip.squeeze(0).numpy()  # shape: (10, 1080, 32)
        # Visualize all 10 frames in the clip
        for i, frame in enumerate(clip):
            plt.figure(figsize=(10, 3))
            plt.imshow(frame.T, cmap='hot', aspect='auto')
            plt.title(f"Frame {i}")
            plt.xlabel("Azimuth bins")
            plt.ylabel("Channels")
            plt.colorbar(label="Distance (mm)")
            plt.show()
        break

def main():
    d_train, d_val, d_test = CreateDOVeDatasets(os.path.abspath('../../Dataset/sample'), frames_per_clip=10, raw=False, granularity=3)
    loader_train = DataLoader(d_train, batch_size=32, shuffle=True, num_workers=3)

    get_dataloader_info(loader_train)
    #visualize_frame(loader_train) 


if __name__ == "__main__":
    main()

