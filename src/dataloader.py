import os
import sys
import cv2
import math
import zipfile
import argparse
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append("./src/")

from utils import (
    dump,
    load,
    config,
)


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
            batch_size=self.batch_size * 4,
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

    @staticmethod
    def display_images():
        dataset = load(
            os.path.join(config()["path"]["PROCESSED_PATH"], "train_dataloader.pkl")
        )

        images, maks = next(iter(dataset))

        number_of_rows = int(math.sqrt(images.size(0)))
        number_of_columns = int(images.size(0) // number_of_rows)

        plt.figure(figsize=(10, 10))

        plt.suptitle("Images and Masks".capitalize())

        for index, image in enumerate(images):
            image = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            mask = maks[index].squeeze().permute(1, 2, 0).detach().cpu().numpy()

            image = (image - image.min()) / (image.max() - image.min())
            mask = (mask - mask.min()) / (mask.max() - mask.min())

            plt.subplot(2 * number_of_rows, 2 * number_of_columns, 2 * index + 1)
            plt.imshow(image)
            plt.title("Image")
            plt.axis("off")

            plt.subplot(2 * number_of_rows, 2 * number_of_columns, 2 * index + 2)
            plt.imshow(mask)
            plt.title("Mask")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(config()["path"]["FILES_PATH"], "image.png"))
        plt.show()

        print(
            "Images saved in the folder: {}".format(
                config()["path"]["FILES_PATH"]
            ).capitalize()
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dataloader for the attentionCNN".capitalize()
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=config()["dataloader"]["image_path"],
        help="Path to the image dataset".capitalize(),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=config()["dataloader"]["image_size"],
        help="Size of the image".capitalize(),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config()["dataloader"]["batch_size"],
        help="Batch size for the dataloader".capitalize(),
    )
    parser.add_argument(
        "--split_size",
        type=float,
        default=config()["dataloader"]["split_size"],
        help="Split size for the dataloader".capitalize(),
    )

    args = parser.parse_args()

    if args.image_path:

        loader = Loader(
            image_path=args.image_path,
            image_size=args.image_size,
            batch_size=args.batch_size,
            split_size=args.split_size,
        )

        # loader.unzip_folder()
        # loader.create_dataloader()

        loader.display_images()

    else:
        raise ValueError("Please provide the path to the image dataset".capitalize())
