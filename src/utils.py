import sys
import yaml
import joblib
import torch

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
        torch.nn.init.kaiming_normal_(m.weight)

    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.constant_(m.bias, 0.0)


def config():
    with open("./config.yml", "r") as file:
        return yaml.safe_load(file)
