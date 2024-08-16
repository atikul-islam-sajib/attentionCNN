import os
import sys
import cv2
import zipfile
import argparse
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.append("./src/")

from utils import dump, config


class Loader:
    def __init__(
        self,
        image_path=None,
        image_size: int = 128,
        batch_size: int = 16,
        split_size: float = 0.2,
    ):
        super(Loader, self).__init__()

        self.image_path = image_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.split_size = split_size

        self.train_images = list()
        self.valid_images = list()

        self.train_masks = list()
        self.valid_masks = list()

    def unzip_folder(self):
        if os.path.exists(self.image_path):
            with zipfile.ZipFile(self.image_path, "r") as zip_file:
                zip_file.extractall(path=config()["path"]["RAW_PATH"])

        else:
            raise FileNotFoundError(
                "Image path not found in the Loader class".capitalize()
            )

    def transforms(self):
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.CenterCrop((self.image_size, self.image_size)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def split_dataset(self, X: list, y: list):
        if isinstance(X, list) and isinstance(y, list):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.split_size, random_state=42
            )

            return {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            }

        else:
            raise TypeError("X and y must be of type list".capitalize())

    def feature_extractor(self):
        self.directory = os.path.join(config()["path"]["RAW_PATH"], "dataset")
        self.train_directory = os.path.join(self.directory, "train")
        self.valid_directory = os.path.join(self.directory, "test")

        for directory in tqdm([self.train_directory, self.valid_directory]):
            self.images = os.path.join(directory, "image")
            self.masks = os.path.join(directory, "mask")

            for image in os.listdir(self.images):
                mask = os.path.join(self.masks, image)
                image = os.path.join(self.images, image)

                image_name = image.split("/")[-1]
                mask_name = mask.split("/")[-1]

                if image_name == mask_name:
                    extracted_image = cv2.imread(image)
                    extracted_mask = cv2.imread(mask)

                    extracted_image = cv2.cvtColor(extracted_image, cv2.COLOR_BGR2RGB)
                    extracted_mask = cv2.cvtColor(extracted_mask, cv2.COLOR_BGR2RGB)

                    extracted_image = Image.fromarray(extracted_image)
                    extracted_mask = Image.fromarray(extracted_mask)

                    extracted_image = self.transforms()(extracted_image)
                    extracted_mask = self.transforms()(extracted_mask)

                    if directory.split("/")[-1] == "train":
                        self.train_images.append(extracted_image)
                        self.train_masks.append(extracted_mask)

                    elif directory.split("/")[-1] == "test":
                        self.valid_images.append(extracted_image)
                        self.valid_masks.append(extracted_mask)

                else:
                    print("Image and mask names do not match".capitalize())

        assert len(self.train_images) == len(
            self.train_masks
        ), "Number of images and masks do not match".capitalize()
        assert len(self.valid_images) == len(
            self.valid_masks
        ), "Number of images and masks do not match".capitalize()

        try:
            dataset = self.split_dataset(X=self.train_images, y=self.train_masks)

        except TypeError as e:
            print("An error occurred while splitting the dataset: ", e)
        except Exception as e:
            print("An error occurred while splitting the dataset: ", e)

        else:
            return dataset, {
                "valid_images": self.valid_images,
                "valid_masks": self.valid_masks,
            }

    def create_dataloader(self):
        train_dataset, valid_dataset = self.feature_extractor()

        train_dataloader = DataLoader(
            dataset=list(zip(train_dataset["X_train"], train_dataset["y_train"])),
            batch_size=self.batch_size,
            shuffle=True,
        )
        test_dataloader = DataLoader(
            dataset=list(zip(train_dataset["X_test"], train_dataset["y_test"])),
            batch_size=self.batch_size,
            shuffle=False,
        )
        valid_datalader = DataLoader(
            dataset=zip(valid_dataset["valid_images"], valid_dataset["valid_masks"]),
            batch_size=self.batch_size,
            shuffle=False,
        )

        for filename, value in [
            ("train_dataloader", train_dataloader),
            ("test_dataloader", test_dataloader),
            ("valid_dataloader", valid_datalader),
        ]:
            dump(
                value=value,
                filename=os.path.join(config()["path"]["PROCESSED_PATH"], filename)
                + ".pkl",
            )

        print("Dataloader is saved in the folder of {}".format("./data/processed/"))


if __name__ == "__main__":
    loader = Loader(image_path="./data/raw/dataset.zip", image_size=128, batch_size=16)
    # loader.unzip_folder()
    loader.create_dataloader()
