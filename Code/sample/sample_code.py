import dove_dataset
from torch.utils.data import DataLoader	

d_train, d_val, d_test = dove_dataset.CreateDOVeDatasets('../../Dataset/sample/Gym-2', frames_per_clip=10, raw=False, granularity=3)
loader_train = DataLoader(d_train, batch_size=32, shuffle=True, num_workers=3)