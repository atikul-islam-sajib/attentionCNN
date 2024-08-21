import os
import sys
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append("./src/")

from utils import load, config, device_init
from attentionCNN import attentionCNN

valid_dataloader = os.path.join("./data/processed", "test_dataloader.pkl")
valid_dataloader = load(valid_dataloader)

device = device_init(device="mps")

model = attentionCNN(
    image_channels=config()["attentionCNN"]["image_channels"],
    image_size=config()["attentionCNN"]["image_size"],
    nheads=config()["attentionCNN"]["nheads"],
    dropout=config()["attentionCNN"]["dropout"],
    num_layers=config()["attentionCNN"]["num_layers"],
    activation=config()["attentionCNN"]["activation"],
    bias=True,
)

model = model.to(device)

state_dict = torch.load("./artifacts/checkpoints/best_model/best_model.pth")
state_dict_load = state_dict["model"]

model.load_state_dict(state_dict_load)

images, mask = next(iter(valid_dataloader))

num_of_rows = int(math.sqrt(images.size(0)))
num_of_cols = images.size(0) // num_of_rows

predicted = model(images.to(device))
mask = mask.to(device)

plt.figure(figsize=(10, 20))

for index, image in enumerate(images):
    real_image = image.permute(1, 2, 0).cpu().detach().numpy()
    predicted_image = predicted[index].permute(1, 2, 0).cpu().detach().numpy()
    mask_image = mask[index].permute(1, 2, 0).cpu().detach().numpy()

    real_image = (real_image - real_image.min()) / (real_image.max() - real_image.min())
    predicted_image = (predicted_image - predicted_image.min()) / (
        predicted_image.max() - predicted_image.min()
    )
    mask_image = (mask_image - mask_image.min()) / (mask_image.max() - mask_image.min())

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
plt.show()
