import os
import sys
import zipfile
import argparse


class Loader:
    def __init__(self, image_path=None, image_size: int = 128, batch_size: int = 16):
        super(Loader, self).__init__()

        self.image_path = image_path
        self.image_size = image_size
        self.batch_size = batch_size

    def unzip_folder(self):
        if os.path.exists(self.image_path):
            with zipfile.ZipFile(self.image_path, "r") as zip_file:
                zip_file.extractall(path="./data/processed/")

        else:
            raise FileNotFoundError(
                "Image path not found in the Loader class".capitalize()
            )


if __name__ == "__main__":
    loader = Loader(image_path="./data/raw/dataset.zip", image_size=128, batch_size=16)
    loader.unzip_folder()
