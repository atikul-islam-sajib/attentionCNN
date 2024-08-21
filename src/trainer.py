import os
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR

sys.path.append("./src/")

from helper import helper
from attentionCNN import attentionCNN
from utils import config, device_init, dump, load


class Trainer:
    def __init__(
        self,
        model=None,
        epochs: int = 100,
        lr: float = 0.0001,
        beta1: float = 0.5,
        beta2: float = 0.999,
        momentum: float = 0.90,
        adam: bool = True,
        SGD: bool = False,
        loss="bce",
        smooth: float = 1e-4,
        alpha: float = 0.25,
        gamma: int = 2,
        step_size: int = 20,
        device: str = "cuda",
        lr_scheduler: bool = False,
        l1_regularization: bool = False,
        l2_regularization: bool = False,
        elasticnet_regularization: bool = False,
        verbose: bool = True,
    ):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentum = momentum
        self.adam = adam
        self.SGD = SGD
        self.loss = loss
        self.smooth = smooth
        self.alpha = alpha
        self.gamma = gamma
        self.step_size = step_size
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization
        self.elasticnet_regularization = elasticnet_regularization
        self.verbose = verbose

        self.device = device_init(
            device=self.device,
        )

        self.init = helper(
            model=self.model,
            lr=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
            momentum=self.momentum,
            adam=self.adam,
            SGD=self.SGD,
            loss=self.loss,
            smooth=self.smooth,
            alpha=self.alpha,
            gamma=self.gamma,
        )

        self.train_dataloader = self.init["train_dataloader"]
        # self.valid_dataloader = self.init["valid_dataloader"]
        self.test_dataloader = self.init["test_dataloader"]

        self.model = self.init["model"]
        self.model = self.model.to(self.device)

        self.optimizer = self.init["optimizer"]
        self.criterion = self.init["loss"]

        assert (
            self.init["train_dataloader"].__class__
        ) == torch.utils.data.dataloader.DataLoader, (
            "train_dataloader is not a dataloader".capitalize()
        )
        # assert (
        #     self.init["valid_dataloader"].__class__
        # ) == torch.utils.data.dataloader.DataLoader, (
        #     "valid_dataloader is not a dataloader".capitalize()
        # )
        assert (
            self.init["test_dataloader"].__class__
        ) == torch.utils.data.dataloader.DataLoader, (
            "test_dataloader is not a dataloader".capitalize()
        )
        assert (
            self.init["model"].__class__
        ) == attentionCNN, "model is not a model".capitalize()
        assert (
            self.init["optimizer"].__class__
        ) == optim.Adam, "optimizer is not an optimizer".capitalize()

        if self.lr_scheduler:
            self.scheduler = StepLR(
                optimizer=self.optimizer, step_size=self.step_size, gamma=self.gamma
            )

        self.loss = float("inf")

        self.model_history = {"train_loss": [], "test_loss": []}

    def l1_loss(self, model=None):
        if isinstance(model, attentionCNN):
            return 0.01 * (torch.norm(params, 1) for params in model.parameters())
        else:
            raise ValueError("model is not a model".capitalize())

    def l2_loss(self, model=None):
        if isinstance(model, attentionCNN):
            return 0.01 * (torch.norm(params, 2) for params in model.parameters())
        else:
            raise ValueError("model is not a model".capitalize())

    def elasticnet_loss(self, model=None):
        if isinstance(model, attentionCNN):
            return 0.01 * (
                torch.norm(params, 1) + torch.norm(params, 2)
                for params in model.parameters()
            )
        else:
            raise ValueError("model is not a model".capitalize())

    def saved_checkpoints(self, **kwargs):
        try:
            epoch = kwargs["epoch"]
            train_loss = kwargs["train_loss"]
            valid_loss = kwargs["valid_loss"]
        except KeyError:
            raise ValueError(
                "Missing required arguments: 'epoch', 'train_loss', or 'valid_loss'"
            )
        else:
            if self.loss > valid_loss:
                self.loss = valid_loss
                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "train_loss": train_loss,
                        "valid_loss": valid_loss,
                        "epoch": epoch,
                    },
                    os.path.join(config()["path"]["BEST_MODEL"], "best_model.pth"),
                )

            torch.save(
                self.model.state_dict(),
                os.path.join(
                    config()["path"]["TRAIN_MODELS"], f"model_epoch_{epoch}.pth"
                ),
            )

    def updated_training_model(self, **kwargs):
        self.optimizer.zero_grad()

        try:
            image = kwargs["image"]
            mask = kwargs["mask"]
        except KeyError:
            raise ValueError("Missing a dataloader".capitalize())

        else:
            predicted = self.model(image)

            loss = self.criterion(predicted, mask)

            if self.l1_regularization:
                loss += self.l1_loss(model=self.model)
            elif self.l2_regularization:
                loss += self.l2_loss(model=self.model)
            elif self.elasticnet_regularization:
                loss += self.elasticnet_loss(predicted, mask)
            else:
                loss = loss

            loss.backward()
            self.optimizer.step()

            return loss.item()

    def saved_training_images(self, **kwargs):
        try:
            epoch = kwargs["epoch"]
        except KeyError:
            raise ValueError("Missing a dataloader".capitalize())
        else:
            images, mask = next(iter(self.test_dataloader))
            images = images.to(self.device)

            predicted = self.model(images)

            save_image(
                predicted,
                os.path.join(
                    config()["path"]["TRAIN_IMAGES"], "image{}.png".format(epoch)
                ),
                normalize=True,
            ),
            save_image(
                mask,
                os.path.join(
                    config()["path"]["TRAIN_IMAGES"], "real_image{}.png".format(epoch)
                ),
                normalize=True,
            ),

    def display_progress(self, **kwargs):
        try:
            epoch = kwargs["epoch"]
            train_loss = kwargs["train_loss"]
            valid_loss = kwargs["valid_loss"]
        except KeyError:
            raise ValueError("Missing a dataloader".capitalize())

        else:
            if self.verbose:
                print(
                    "Epochs: [{}/{}] - train_loss: {:.4f} - valid_loss: {:.4f}".format(
                        epoch, self.epochs, train_loss, valid_loss
                    )
                )
            else:
                print(
                    "Epochs: [{}/{}] is completed".format(
                        epoch, self.epochs
                    ).capitalize()
                )

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            train_loss = []
            valid_loss = []

            for _, (image, mask) in enumerate(self.train_dataloader):
                image = image.to(self.device)
                mask = mask.to(self.device)

                try:
                    train_loss.append(
                        self.updated_training_model(image=image, mask=mask)
                    )
                except KeyError as e:
                    print("An error occured: {}".format(e))
                except Exception as e:
                    print("An error occured: {}".format(e))

            for _, (image, mask) in enumerate(self.test_dataloader):
                image = image.to(self.device)
                mask = mask.to(self.device)

                predicted = self.model(image)

                valid_loss.append(self.criterion(predicted, mask).item())

            if self.lr_scheduler:
                self.scheduler.step()

            try:
                self.display_progress(
                    epoch=epoch + 1,
                    train_loss=np.mean(train_loss),
                    valid_loss=np.mean(valid_loss),
                )
            except KeyError as e:
                print("An error occured: {}".format(e).capitalize())
            except Exception as e:
                print("An error occured: {}".format(e).capitalize())

            try:
                self.saved_training_images(epoch=epoch)
            except KeyError as e:
                print("An error occured: {}".format(e).capitalize())
            except Exception as e:
                print("An error occured: {}".format(e).capitalize())

            try:
                self.saved_training_images(epoch=epoch)
            except KeyError as e:
                print("An error occured: {}".format(e).capitalize())
            except Exception as e:
                print("An error occured: {}".format(e).capitalize())

            try:
                self.saved_checkpoints(
                    epoch=epoch,
                    train_loss=np.mean(train_loss),
                    valid_loss=np.mean(valid_loss),
                )
            except KeyError as e:
                print("An error occured in : {}".format(e).capitalize())
            except Exception as e:
                print("An error occuredin in: {}".format(e).capitalize())

            try:
                self.model_history["train_loss"].append(np.mean(train_loss))
                self.model_history["test_loss"].append(np.mean(valid_loss))
            except Exception as e:
                print("An error occured: {}".format(e).capitalize())

        dump(
            value=self.model_history,
            filename=os.path.join(config()["path"]["METRICS_PATH"], "history.pkl"),
        )
        print(
            "Model history saved in the folder {}".format(
                config()["path"]["METRICS_PATH"]
            ).capitalize()
        )

    @staticmethod
    def display_history():
        metrics_path = config()["path"]["METRICS_PATH"]
        history = load(filename=os.path.join(metrics_path, "history.pkl"))

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        fig.suptitle("Loss Evolution")

        axes[0].plot(history["train_loss"], label="Train Loss", color="b")
        axes[1].plot(history["test_loss"], label="Test Loss", color="r")

        axes[0].grid(True, which="both", linestyle="--", linewidth=0.5)
        axes[1].grid(True, which="both", linestyle="--", linewidth=0.5)

        axes[0].set_title("Train Loss")
        axes[1].set_title("Test Loss")

        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Loss")

        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Loss")

        axes[0].legend()
        axes[1].legend()

        plt.legend()
        plt.savefig(os.path.join(metrics_path, "loss_evolution.png"))
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the attentionCNN model".title())
    parser.add_argument(
        "--epochs",
        type=int,
        default=config()["model"]["epochs"],
        help="number of epochs to train the model".capitalize(),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config()["model"]["lr"],
        help="learning rate of the model".capitalize(),
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=config()["model"]["momentum"],
        help="momentum of the model".capitalize(),
    )
    parser.add_argument(
        "--adam",
        type=bool,
        default=config()["model"]["adam"],
        help="adam optimizer of the model".capitalize(),
    )
    parser.add_argument(
        "--SGD",
        type=bool,
        default=config()["model"]["SGD"],
        help="SGD optimizer of the model".capitalize(),
    )
    parser.add_argument(
        "--loss",
        type=str,
        default=config()["model"]["loss"],
        help="loss function of the model".capitalize(),
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=config()["model"]["smooth"],
        help="smooth loss function of the model".capitalize(),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=config()["model"]["alpha"],
        help="alpha of the model".capitalize(),
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=config()["model"]["beta1"],
        help="beta of the model".capitalize(),
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=config()["model"]["beta2"],
        help="beta of the model".capitalize(),
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=config()["model"]["step_size"],
        help="step size of the model".capitalize(),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config()["model"]["device"],
        help="device of the model".capitalize(),
    )
    parser.add_argument(
        "--lr_scheduler",
        type=bool,
        default=config()["model"]["lr_scheduler"],
        help="lr scheduler of the model".capitalize(),
    )
    parser.add_argument(
        "--l1_regularization",
        type=bool,
        default=config()["model"]["l1_regularization"],
        help="l1 regularization of the model".capitalize(),
    )
    parser.add_argument(
        "--l2_regularization",
        type=bool,
        default=config()["model"]["l2_regularization"],
        help="l2 regularization of the model".capitalize(),
    )
    parser.add_argument(
        "--elasticnet_regularization",
        type=bool,
        default=config()["model"]["elasticnet_regularization"],
        help="elasticnet regularization of the model".capitalize(),
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=config()["model"]["verbose"],
        help="verbose of the model".capitalize(),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=config()["model"]["gamma"],
        help="gamma of the model".capitalize(),
    )

    args = parser.parse_args()

    trainer = Trainer(
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        loss=args.loss,
        beta1=args.beta1,
        beta2=args.beta2,
        lr_scheduler=args.lr_scheduler,
        l1_regularization=args.l1_regularization,
        l2_regularization=args.l2_regularization,
        elasticnet_regularization=args.elasticnet_regularization,
        step_size=args.step_size,
        gamma=args.gamma,
        alpha=args.alpha,
        smooth=args.smooth,
        momentum=args.momentum,
        adam=args.adam,
        SGD=args.SGD,
        verbose=args.verbose,
        model=None,
    )

    trainer.train()

    Trainer.display_history()
