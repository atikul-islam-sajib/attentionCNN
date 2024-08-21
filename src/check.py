import os
from utils import config, load

valid = load(os.path.join(config()["path"]["PROCESSED_PATH"], "valid_dataloader.pkl"))

print(next(iter(valid))[0].shape)
