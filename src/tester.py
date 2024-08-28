import os
import sys
import math
import torch
import argparse
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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

    def compute_iou(self, predicted_mask, real_mask, threshold=0.4):
        """
        Compute the Intersection over Union (IoU) between the predicted mask and the real mask using PyTorch.

        Parameters:
        - predicted_mask: A 2D or 3D PyTorch tensor with predicted mask values (continuous or binary).
        - real_mask: A 2D or 3D PyTorch tensor with the real mask values (binary).
        - threshold: A float value to threshold the predicted mask if it contains continuous values.

        Returns:
        - iou: The IoU score as a float.
        """
        # # Apply threshold to predictions to convert them to binary
        # y_pred = (predicted_mask >= threshold).to(torch.int)
        # y_true = real_mask.to(torch.int)

        # # Flatten the tensors
        # y_pred_flat = y_pred.view(y_pred.size(0), -1)
        # y_true_flat = y_true.view(y_true.size(0), -1)

        # # Initialize an array to store IoU for each image in the batch
        # iou_scores = []

        # for i in range(y_pred.size(0)):
        #     # Compute confusion matrix for each image in the batch
        #     current = confusion_matrix(
        #         y_true_flat[i].cpu().numpy(),
        #         y_pred_flat[i].cpu().numpy(),
        #         labels=[0, 1],
        #     )

        #     # Intersection: Diagonal of the confusion matrix (True Positives for each class)
        #     intersection = np.diag(current)

        #     # Ground truth set (sum of each row in the confusion matrix)
        #     ground_truth_set = current.sum(axis=1)

        #     # Predicted set (sum of each column in the confusion matrix)
        #     predicted_set = current.sum(axis=0)

        #     # Union: Ground truth + predicted - intersection
        #     union = ground_truth_set + predicted_set - intersection

        #     # IoU for each class
        #     IoU = intersection / union.astype(np.float32)

        #     # Store the mean IoU for this image
        #     iou_scores.append(np.mean(IoU))

        # # Calculate the mean IoU across the batch
        # mean_iou = np.mean(iou_scores)

        # return mean_iou

        binary_pred_masks = (predicted_mask >= threshold).to(torch.bool)

        # Ensure real_masks are binary and convert to boolean
        binary_real_masks = (real_mask >= threshold).to(torch.bool)

        # Calculate intersection and union for each image in the batch
        intersection = torch.logical_and(binary_pred_masks, binary_real_masks).sum(
            dim=(1, 2, 3)
        )
        union = torch.logical_or(binary_pred_masks, binary_real_masks).sum(
            dim=(1, 2, 3)
        )

        # Calculate IoU for each image in the batch
        iou = torch.where(
            union == 0,
            torch.ones_like(intersection, dtype=torch.float),
            intersection.float() / union.float(),
        )

        # Calculate the mean IoU across the batch
        mean_iou = iou.mean().item()

        return mean_iou

    def compute_dice_score(self, y_pred, y_true, threshold=0.5):
        """
        Compute the Dice Score for a batch of predicted masks and real masks using PyTorch.

        Parameters:
        - y_pred: A 4D PyTorch tensor with predicted mask values (batch_size, channels, height, width).
        - y_true: A 4D PyTorch tensor with the real mask values (binary, same dimensions as y_pred).
        - threshold: A float value to threshold the predicted masks if they contain continuous values.

        Returns:
        - mean_dice: The average Dice score across the batch as a float.
        """

        # Apply threshold to predictions to convert them to binary
        y_pred = (y_pred >= threshold).float()
        y_true = y_true.float()

        # Flatten the tensors to compute the Dice score across each image
        y_pred_flat = y_pred.view(y_pred.size(0), -1)
        y_true_flat = y_true.view(y_true.size(0), -1)

        # Calculate intersection and union
        intersection = (y_pred_flat * y_true_flat).sum(dim=1)
        union = y_pred_flat.sum(dim=1) + y_true_flat.sum(dim=1)

        # Compute the Dice score for each image in the batch
        dice = 2 * intersection / union

        # Calculate the mean Dice score across the batch
        mean_dice = dice.mean().item()

        return 1 - mean_dice

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

            IOU = self.compute_iou(predicted_mask=predicted, real_mask=mask)

            print(IOU)

            DICE = self.compute_dice_score(y_pred=predicted, y_true=mask)

            print(DICE)

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

            print(
                "Saved the result image to {}".format(
                    os.path.join(
                        config()["path"]["TEST_IMAGE"],
                        "{}_result.png".format(self.data),
                    )
                )
            )


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
