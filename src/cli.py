import sys
import argparse
import traceback

sys.path.append("./src/")

from utils import config
from tester import Tester
from trainer import Trainer
from dataloader import Loader


def cli():
    parser = argparse.ArgumentParser(
        description="CLI for the attentionCNN".capitalize()
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=config()["Trainer"]["epochs"],
        help="number of epochs to train the model".capitalize(),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config()["Trainer"]["lr"],
        help="learning rate of the model".capitalize(),
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=config()["Trainer"]["momentum"],
        help="momentum of the model".capitalize(),
    )
    parser.add_argument(
        "--adam",
        type=bool,
        default=config()["Trainer"]["adam"],
        help="adam optimizer of the model".capitalize(),
    )
    parser.add_argument(
        "--SGD",
        type=bool,
        default=config()["Trainer"]["SGD"],
        help="SGD optimizer of the model".capitalize(),
    )
    parser.add_argument(
        "--loss",
        type=str,
        default=config()["Trainer"]["loss"],
        help="loss function of the model".capitalize(),
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=config()["Trainer"]["smooth"],
        help="smooth loss function of the model".capitalize(),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=config()["Trainer"]["alpha"],
        help="alpha of the model".capitalize(),
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=config()["Trainer"]["beta1"],
        help="beta of the model".capitalize(),
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=config()["Trainer"]["beta2"],
        help="beta of the model".capitalize(),
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=config()["Trainer"]["step_size"],
        help="step size of the model".capitalize(),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config()["Trainer"]["device"],
        help="device of the model".capitalize(),
    )
    parser.add_argument(
        "--lr_scheduler",
        type=bool,
        default=config()["Trainer"]["lr_scheduler"],
        help="lr scheduler of the model".capitalize(),
    )
    parser.add_argument(
        "--l1_regularization",
        type=bool,
        default=config()["Trainer"]["l1_regularization"],
        help="l1 regularization of the model".capitalize(),
    )
    parser.add_argument(
        "--l2_regularization",
        type=bool,
        default=config()["Trainer"]["l2_regularization"],
        help="l2 regularization of the model".capitalize(),
    )
    parser.add_argument(
        "--elasticnet_regularization",
        type=bool,
        default=config()["Trainer"]["elasticnet_regularization"],
        help="elasticnet regularization of the model".capitalize(),
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=config()["Trainer"]["verbose"],
        help="verbose of the model".capitalize(),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=config()["Trainer"]["gamma"],
        help="gamma of the model".capitalize(),
    )
    parser.add_argument(
        "--data",
        type=str,
        default=config()["Tester"]["data"],
        choices=["test", "valid"],
        help="data type: train, valid, test".capitalize(),
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="train the model".capitalize(),
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="test the model".capitalize(),
    )
    parser.add_argument(
        "--mlflow",
        type=bool,
        default=config()["Trainer"]["mlflow"],
        help="mlflow of the model".capitalize(),
    )
    parser.add_argument(
        "--weight_init",
        type=bool,
        default=config()["Trainer"]["weight_init"],
        help="Weight initialization of the model".capitalize(),
    )

    args = parser.parse_args()

    if args.train:
        loader = Loader(
            image_path=args.image_path,
            image_size=args.image_size,
            batch_size=args.batch_size,
            split_size=args.split_size,
        )

        try:
            loader.unzip_folder()
            loader.create_dataloader()

            loader.display_images()
        except KeyError as e:
            print("Please provide a valid path to the images".capitalize())
            traceback.print_exc()
        except ValueError as e:
            print("An error occurred: {}".format(e))
            traceback.print_exc()

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
            is_mlflow=args.mlflow,
            is_weight_init=args.weight_init,
            model=None,
        )

        try:
            trainer.train()
        except KeyError as e:
            print("An error occured while training the model", e)
            traceback.print_exc()
        except ValueError as e:
            print("An error occured while training the model", e)
            traceback.print_exc()
        except Exception as e:
            print("An error occured while training the model", e)
            traceback.print_exc()
        else:
            Trainer.display_history()

    elif args.test:
        tester = Tester(data=args.data, device=args.device)

        try:
            tester.plot_images()
        except ValueError as e:
            print("An error is occurred: {}".format(e))
        except Exception as e:
            print("An error is occurred: {}".format(e))

    else:
        raise ValueError("Please specify a valid action: train or test".capitalize())


if __name__ == "__main__":
    cli()
