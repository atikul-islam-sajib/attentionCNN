# attentionCNN

This project utilizes a customized Transformer architecture where the standard Multi-Layer Perceptron (MLP) within the Transformer encoder layer has been replaced with a Convolutional Layer (Conv2D). This modification aims to leverage the spatial hierarchies that Conv2D layers capture, providing a richer representation of the input data. 

After processing the data through the Transformer encoder with Conv2D layers, an Autoencoder is employed. The Autoencoder is responsible for further compressing the output, learning efficient representations, and reconstructing the input. This combination of a Transformer with Conv2D and Autoencoder provides a powerful approach for tasks that require both sequential modeling and spatial feature extraction.

<img src="https://github.com/atikul-islam-sajib/Research-Assistant-Work-/blob/main/IMG_0772.jpg" alt="attentionCNN Result">


<img src="https://github.com/atikul-islam-sajib/attentionCNN/blob/main/artifacts/files/valid_result.png" alt="attentionCNN Result">

## Installation

To get started, clone the repository and install the required dependencies. It is recommended to use a virtual environment to manage dependencies.

### Clone the Repository

```bash
git clone https://github.com/atikul-islam-sajib/attentionCNN.git
cd attentionCNN
```

### Set Up the Environment

```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
pip install -r requirements.txt
```

## Project Structure
```
.
├── Dockerfile
├── LICENSE
├── README.md
├── artifacts
│   ├── checkpoints
│   │   ├── best_model
│   │   │   └── best_model.pth
│   │   └── train_models
│   ├── files
│   ├── metrics
│   └── outputs
│       ├── test_image
│       └── train_images
│       └── dataset.zip
├── dvc.lock
├── dvc.yaml
├── logs
│   └── _.py
├── mlruns
│   ├── 0
├── mypy.ini
├── notebooks
│   ├── Inference.ipynb
│   └── ModelPrototype.ipynb
├── requirements.txt
├── setup.py
├── src
│   ├── __init__.py
│   ├── attentionCNN.py
│   ├── attentionCNNBlock.py
│   ├── cli.py
│   ├── dataloader.py
│   ├── decoder_block.py
│   ├── encoder_block.py
│   ├── feedforward_network.py
│   ├── helper.py
│   ├── loss
│   │   ├── __init__.py
│   │   ├── bce_loss.py
│   │   ├── combo_loss.py
│   │   ├── dice_loss.py
│   │   ├── focal_loss.py
│   │   ├── jaccard_loss.py
│   │   ├── mse_loss.py
│   │   └── tversky_loss.py
│   ├── multihead_attention.py
│   ├── scaled_dot_product.py
│   ├── tester.py
│   ├── trainer.py
│   └── utils.py
└── unittest
    └── test.py
```

## Dataset Structure:

```
dataset/  # Folder name must be 'dataset'
├── train/
│   ├── 2.png
│   ├── 3.png
│   ├── ...
├── test/
│   ├── 2.png
│   ├── 3.png
│   ├── ...
```

### User Guide Notebook (Tutorial for inferencing)

For detailed documentation on the implementation and usage, visit the -> [attentionCNN Tutorial Notebook](https://github.com/atikul-islam-sajib/attentionCNN/blob/main/notebooks/AttentionCNN_Tutorial.ipynb)

### Important Notes:

- The folder name must be `dataset`.
- Each `folder*` (e.g., `train`, `test`, etc.) will contain the image data.
- Inside each `folder*`, the images are named sequentially (e.g., `2.png`, `3.png`, `4.png`, `5.png`, etc.).
- The structure is designed to group related images within each folder, which may represent different categories, classes, or datasets for a specific purpose.
- Ensure that all image files are correctly named and placed in their respective folders to maintain the integrity and organization of the dataset.

**Configure the Project**:
```yaml 
path:
  RAW_PATH: "./data/raw/"                                  # Path to raw data
  PROCESSED_PATH: "./data/processed/"                      # Path to processed data
  FILES_PATH: "./artifacts/files/"                         # Path to store files and artifacts
  TRAIN_IMAGES: "./artifacts/outputs/train_images/"        # Path to save training images
  TEST_IMAGE: "./artifacts/outputs/test_image/"            # Path to save a test image
  TRAIN_MODELS: "./artifacts/checkpoints/train_models/"    # Path to save trained models
  BEST_MODEL: "./artifacts/checkpoints/best_model/"        # Path to save the best model checkpoint
  METRICS_PATH: "./artifacts/metrics/"                     # Path to save metrics and evaluation results

dataloader:
  image_path: "./data/raw/dataset.zip"                     # Path to the image dataset (zip file)
  image_size: 128                                          # Size to which images will be resized
  batch_size: 16                                           # Batch size for loading data
  split_size: 0.20                                         # Proportion of data used for validation/testing

attentionCNN:
  image_channels: 3                                        # Number of channels in the input images (e.g., 3 for RGB)
  image_size: 128                                          # Size of the input images
  nheads: 8                                                # Number of attention heads in the Transformer layer
  dropout: 0.3                                             # Dropout rate for regularization
  num_layers: 2                                            # Number of layers in the Transformer encoder
  activation: "relu"                                       # Activation function used in the network
  bias: True                                               # Whether to use bias in the network layers

MLFlow:
  MLFLOW_TRACKING_URI: "https://github.com/atikul-islam-sajib/attentionCNN.git"     # URI for MLflow tracking
  MLFLOW_EXPERIMENT_NAME: "attentionCNN - last training"                            # Experiment name in MLflow
  MLFLOW_USERNAME: "atikul-islam-sajib"                                             # Username for MLflow access
  MLFLOW_REPONAME: "attentionCNN"                                                   # Repository name for MLflow
  MLFLOW_PASSWORD: "*****************************************"                      # Password/token for MLflow authentication

Trainer:
  epochs: 500                                             # Number of epochs to train the model
  lr: 0.0001                                              # Learning rate for training
  beta1: 0.75                                             # Beta1 parameter for the Adam optimizer
  beta2: 0.999                                            # Beta2 parameter for the Adam optimizer
  momentum: 0.95                                          # Momentum factor for SGD optimizer
  smooth: 1e-4                                            # Smoothing factor for the loss function
  alpha: 0.25                                             # Alpha value (e.g., for focal loss or regularization)
  gamma: 2                                                # Gamma value (e.g., for learning rate scheduler)
  step_size: 20                                           # Step size for learning rate scheduler
  weight_decay: 0.001                                     # Weight decay (L2 penalty) for regularization
  device: "mps"                                           # Device to use for training (e.g., cpu, cuda, mps)
  loss: "mse"                                             # Loss function to use (e.g., MSE)
  adam: True                                              # Use Adam optimizer if True
  SGD: False                                              # Use SGD optimizer if True
  lr_scheduler: False                                     # Use learning rate scheduler if True
  weight_init: True                                       # Initialize model weights if True
  l1_regularization: False                                # Apply L1 regularization if True
  l2_regularization: False                                # Apply L2 regularization if True
  elasticnet_regularization: False                        # Apply Elastic Net regularization if True
  mlflow: True                                            # Track experiments with MLflow if True
  verbose: True                                           # Enable verbose output if True

Tester:
  data: "test"                                            # Specify which dataset to use for testing (test or valid)
  device: "mps"                                           # Device to use for testing (e.g., cpu, cuda, mps)
```

### Configuration for MLFlow

1. **Generate a Personal Access Token on DagsHub**:
   - Log in to [DagsHub](https://dagshub.com).
   - Go to your user settings and generate a new personal access token under "Personal Access Tokens".


2. **Configuration in config.yml**:
   Ensure the MLFlow configuration is defined in the `config.yml` file. The relevant section might look like this:

   ```yaml
   MLFlow:
     MLFLOW_TRACKING_URL: "https://dagshub.com/<username>/<repo_name>.mlflow"
     MLFLOW_TRACKING_USERNAME: "<your_dagshub_username>"
     MLFLOW_TRACKING_PASSWORD: "<your_dagshub_token>"
   ```

   Make sure to replace `<username>`, `<repo_name>`, `<your_dagshub_username>`, and `<your_dagshub_token>` with your actual DagsHub credentials.

### Command Line Interface

| Argument                  | Type     | Default Value                     | Description                                                  |
|---------------------------|----------|-----------------------------------|--------------------------------------------------------------|
| `--image_path`            | `str`    | `config()["dataloader"]["image_path"]` | Path to the image dataset                                    |
| `--image_size`            | `int`    | `config()["dataloader"]["image_size"]` | Size of the images                                           |
| `--batch_size`            | `int`    | `config()["dataloader"]["batch_size"]` | Batch size for the dataloader                                |
| `--split_size`            | `float`  | `config()["dataloader"]["split_size"]` | Split size for the dataloader                                |
| `--epochs`                | `int`    | `config()["Trainer"]["epochs"]`   | Number of epochs to train the model                          |
| `--lr`                    | `float`  | `config()["Trainer"]["lr"]`       | Learning rate of the model                                   |
| `--momentum`              | `float`  | `config()["Trainer"]["momentum"]` | Momentum for the SGD optimizer                               |
| `--adam`                  | `bool`   | `config()["Trainer"]["adam"]`     | Use Adam optimizer if `True`                                 |
| `--SGD`                   | `bool`   | `config()["Trainer"]["SGD"]`      | Use SGD optimizer if `True`                                  |
| `--loss`                  | `str`    | `config()["Trainer"]["loss"]`     | Loss function to use                                         |
| `--smooth`                | `float`  | `config()["Trainer"]["smooth"]`   | Smoothing factor for loss function                           |
| `--alpha`                 | `float`  | `config()["Trainer"]["alpha"]`    | Alpha value (e.g., for regularization)                       |
| `--beta1`                 | `float`  | `config()["Trainer"]["beta1"]`    | Beta1 parameter for Adam optimizer                           |
| `--beta2`                 | `float`  | `config()["Trainer"]["beta2"]`    | Beta2 parameter for Adam optimizer                           |
| `--step_size`             | `int`    | `config()["Trainer"]["step_size"]`| Step size for learning rate scheduler                        |
| `--device`                | `str`    | `config()["Trainer"]["device"]`   | Device to run the model on (`cpu` or `cuda`)                 |
| `--lr_scheduler`          | `bool`   | `config()["Trainer"]["lr_scheduler"]` | Use learning rate scheduler if `True`                    |
| `--l1_regularization`     | `bool`   | `config()["Trainer"]["l1_regularization"]` | Apply L1 regularization if `True`                    |
| `--l2_regularization`     | `bool`   | `config()["Trainer"]["l2_regularization"]` | Apply L2 regularization if `True`                    |
| `--elasticnet_regularization` | `bool` | `config()["Trainer"]["elasticnet_regularization"]` | Apply Elastic Net regularization if `True`    |
| `--verbose`               | `bool`   | `config()["Trainer"]["verbose"]`  | Verbose mode if `True`                                       |
| `--gamma`                 | `float`  | `config()["Trainer"]["gamma"]`    | Gamma parameter for learning rate scheduler                  |
| `--data`                  | `str`    | `config()["Tester"]["data"]`      | Data type for testing (`test` or `valid`)                    |
| `--train`                 | `store_true` | `None`                        | Train the model                                              |
| `--test`                  | `store_true` | `None`                        | Test the model                                               |
| `--mlflow`                | `bool`   | `config()["Trainer"]["mlflow"]`   | Track experiment with MLflow if `True`                       |
| `--weight_init`           | `bool`   | `config()["Trainer"]["weight_init"]` | Initialize model weights if `True`                       |

This table provides a clear overview of the command-line arguments that can be passed to the script, along with their descriptions, types, and default values.


| **Action**       | **Command Line**                                                                                                                                                                                 |
|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Train the Model** | *python cli.py --train --image_path /path/to/dataset --image_size 224 --batch_size 32 --epochs 50 --lr 0.001 --momentum 0.9 --adam True --loss mse --device cuda --verbose True --mlflow True --weight_init True* |
| **Test the Model**  | *python cli.py --test --data test --device cuda --verbose True*                                                                                                                |

**Explanation:**
- Adjust the paths, hyperparameters, and other options (like `image_size`, `batch_size`, `epochs`, etc.) according to your specific use case.
- For training, ensure that `--train` is included in the command line, along with all the necessary hyperparameters.
- For testing, ensure that `--test` is included, and specify the dataset (`test` or `valid`) using `--data`.

### Accessing Experiment Tracking

You can access the MLflow experiment tracking UI hosted on DagsHub using the following link:

[attentionCNN Experiment Tracking on DagsHub](https://dagshub.com/atikul-islam-sajib/attentionCNN/experiments)

### Using MLflow UI Locally

If you prefer to run the MLflow UI locally, use the following command:

```bash
mlflow ui
```

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

