import os
import sys
import math
import torch
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append("./src/")

from utils import load, config, device_init
from attentionCNN import attentionCNN


class Tester:
    def __init__(self, data: str = "test", device: str = "cuda"):
        self.data = data
        self.device = device

        self.device = device_init(
            device=self.device,
        )

    def load_dataloader(self):
        if self.data == "test":
            test_dataloader = os.path.join(
                config()["path"]["PROCESSED_PATH"], "test_dataloader.pkl"
            )

            test_dataloader = load(filename=test_dataloader)

            return test_dataloader

        elif self.data == "valid":
            valid_dataloader = os.path.join(
                config()["path"]["PROCESSED_PATH"], "valid_dataloader.pkl"
            )
            valid_dataloader = load(filename=valid_dataloader)

            return valid_dataloader

        else:
            raise ValueError("Invalid data type".capitalize())

    def select_model(self):
        try:
            model = attentionCNN(
                image_channels=config()["attentionCNN"]["image_channels"],
                image_size=config()["attentionCNN"]["image_size"],
                nheads=config()["attentionCNN"]["nheads"],
                dropout=config()["attentionCNN"]["dropout"],
                num_layers=config()["attentionCNN"]["num_layers"],
                activation=config()["attentionCNN"]["activation"],
                bias=config()["attentionCNN"]["bias"],
            )

        except Exception as e:
            print("An error occurred to load the model: ", e)
        else:
            return model

    def plot_images(self):
        try:
            model = self.select_model()
        except Exception as e:
            print("An error occurred to load the model: ", e)
        else:
            model = model.to(self.device)

            state_dict = torch.load(
                os.path.join(config()["path"]["BEST_MODEL"], "best_model.pth")
            )

            model.load_state_dict(state_dict["model"])

            images, mask = next(iter(self.load_dataloader()))

            num_of_rows = int(math.sqrt(images.size(0)))
            num_of_cols = images.size(0) // num_of_rows

            predicted = model(images.to(self.device))
            mask = mask.to(self.device)

            plt.figure(figsize=(num_of_rows * 10, num_of_cols * 5))

            for index, image in enumerate(images):
                real_image = image.permute(1, 2, 0).cpu().detach().numpy()
                predicted_image = (
                    predicted[index].permute(1, 2, 0).cpu().detach().numpy()
                )
                mask_image = mask[index].permute(1, 2, 0).cpu().detach().numpy()

                real_image = (real_image - real_image.min()) / (
                    real_image.max() - real_image.min()
                )
                predicted_image = (predicted_image - predicted_image.min()) / (
                    predicted_image.max() - predicted_image.min()
                )
                mask_image = (mask_image - mask_image.min()) / (
                    mask_image.max() - mask_image.min()
                )

                plt.subplot(3 * num_of_rows, 2 * num_of_cols, 3 * index + 1)
                plt.imshow(real_image)
                plt.axis("off")
                plt.title("Real Image")

                plt.subplot(3 * num_of_rows, 2 * num_of_cols, 3 * index + 2)
                plt.imshow(predicted_image, cmap="gray")
                plt.axis("off")
                plt.title("Predicted Image")

                plt.subplot(3 * num_of_rows, 2 * num_of_cols, 3 * index + 3)
                plt.imshow(mask_image, cmap="gray")
                plt.axis("off")
                plt.title("Mask Image")

            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    config()["path"]["TEST_IMAGE"],
                    (
                        "{}_result.png".format(self.data)
                        if self.data == "test"
                        else "{}_result.png".format(self.data)
                    ),
                )
            )
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tester code for attentionCNN".title())
    parser.add_argument(
        "--data",
        type=str,
        default=config()["Tester"]["data"],
        choices=["test", "valid"],
        help="data type: train, valid, test".capitalize(),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config()["Tester"]["device"],
        choices=["cpu", "mps", "cuda"],
        help="device type: cpu, mps, cuda".capitalize(),
    )
    args = parser.parse_args()

    tester = Tester(data=args.data, device=args.device)

    try:
        tester.plot_images()
    except ValueError as e:
        print("An error is occurred: {}".format(e))
    except Exception as e:
        print("An error is occurred: {}".format(e))
