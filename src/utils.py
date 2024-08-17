import sys
import yaml
import joblib

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


def config():
    with open("./config.yml", "r") as file:
        return yaml.safe_load(file)
