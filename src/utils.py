import os
import sys
import yaml
import joblib
import torch
import traceback

sys.path.append("./src/")


def dump(value=None, filename=None):
    if (value is not None) and (filename is not None):
        joblib.dump(value=value, filename=filename)

    else:
        raise ValueError("Both value and filename are required".capitalize())


def load(filename=None):
    if filename is not None:
        return joblib.load(filename=filename)

    else:
        raise ValueError("Filename is required".capitalize())


def device_init(device="cuda"):
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        return torch.device("cpu")


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def config():
    with open("./config.yml", "r") as file:
        return yaml.safe_load(file)


def clean():
    config_files = config()

    RAW_PATH = config_files["path"]["RAW_PATH"]
    PROCESSED_PATH = config_files["path"]["PROCESSED_PATH"]
    FILES_PATH = config_files["path"]["FILES_PATH"]
    TRAIN_IMAGES = config_files["path"]["TRAIN_IMAGES"]
    TEST_IMAGE = config_files["path"]["TEST_IMAGE"]
    TRAIN_MODELS = config_files["path"]["TRAIN_MODELS"]
    BEST_MODEL = config_files["path"]["BEST_MODEL"]
    METRICS_PATH = config_files["path"]["METRICS_PATH"]

    for path in [
        RAW_PATH,
        PROCESSED_PATH,
        FILES_PATH,
        TRAIN_IMAGES,
        TEST_IMAGE,
        TRAIN_MODELS,
        BEST_MODEL,
        METRICS_PATH,
    ]:
        if os.path.exists(path):
            for file in os.listdir(path):
                os.remove(path=os.path.join(path, file))

            print(f"Deleted all files in {path}".capitalize())

        else:
            raise FileNotFoundError(f"{path} does not exist".capitalize())
