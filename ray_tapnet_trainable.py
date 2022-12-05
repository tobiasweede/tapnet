# Original Code here:
# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function

import argparse
import os
import torch
import torch.optim as optim

import ray
from ray import air, tune
from ray.tune.schedulers import ASHAScheduler

from tapnet import TapNet
from utils import boolean_string

# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 512
TEST_SIZE = 256


def parse_args():
    """Parse config args"""
    parser = argparse.ArgumentParser()

    # dataset settings
    parser.add_argument(
        "--load_custom_ts",
        type=boolean_string,
        default=True,
        help="use custom data loader function",
    )
    parser.add_argument(
        "--data_path", type=str, default="./data/", help="the path of data."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",  # NATOPS
        help="time series dataset. Options: See the datasets list",
    )

    # cuda settings
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="Disables CUDA training.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Training parameter settings
    # parser.add_argument(
    #     "--epochs", type=int, default=3000, help="Number of epochs to train."
    # )
    # parser.add_argument(
    #     "--lr",
    #     type=float,
    #     default=1e-5,
    #     help="Initial learning rate. default:[0.00001]",
    # )
    # parser.add_argument(
    #     "--wd",
    #     type=float,
    #     default=1e-3,
    #     help="Weight decay (L2 loss on parameters). default: 5e-3",
    # )
    # parser.add_argument(
    #     "--stop_thres",
    #     type=float,
    #     default=1e-9,
    #     help="The stop threshold for the training error. If the difference between training losses "
    #     "between epochs are less than the threshold, the training will be stopped. Default:1e-9",
    # )

    # Model parameters
    parser.add_argument(
        "--use_cnn",
        type=boolean_string,
        default=True,
        help="whether to use CNN for feature extraction. Default:False",
    )
    parser.add_argument(
        "--use_lstm",
        type=boolean_string,
        default=True,
        help="whether to use LSTM for feature extraction. Default:False",
    )
    parser.add_argument(
        "--use_rp",
        type=boolean_string,
        default=True,
        help="Whether to use random projection",
    )
    parser.add_argument(
        "--rp_params",
        type=str,
        default="-1,3",
        help="Parameters for random projection: number of random projection, "
        "sub-dimension for each random projection",
    )
    parser.add_argument(
        "--use_metric",
        action="store_true",
        default=False,
        help="whether to use the metric learning for class representation. Default:False",
    )
    parser.add_argument(
        "--metric_param",
        type=float,
        default=0.000001,
        help="Metric parameter for prototype distances between classes. Default:0.000001",
    )
    parser.add_argument(
        "--filters",
        type=str,
        default="256,256,128",
        help="filters used for convolutional network. Default:256,256,128",
    )
    parser.add_argument(
        "--kernels",
        type=str,
        default="8,5,3",
        help="kernels used for convolutional network. Default:8,5,3",
    )
    parser.add_argument(
        "--dilation",
        type=int,
        default=-1,
        help="the dilation used for the first convolutional layer. "
        "If set to -1, use the automatic number. Default:-1",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="500,300",
        help="layer settings of mapping function. [Default]: 500,300",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate (1 - keep probability). Default:0.5",
    )
    parser.add_argument(
        "--lstm_dim",
        type=int,
        default=128,
        help="Dimension of LSTM Embedding.",
    )

    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
        help="enables CUDA training",
    )
    parser.add_argument(
        "--ray-address", type=str, help="The Redis address of the cluster."
    )
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    return parser.parse_args()


class TrainTapNet(tune.Trainable):
    def setup(self, config):
        use_cuda = config.get("use_gpu") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.train_loader, self.test_loader = get_data_loaders()
        self.model = ConvNet().to(self.device)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.get("lr", 0.01),
            momentum=config.get("momentum", 0.9),
        )

    def step(self):
        pass

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


if __name__ == "__main__":
    args = parse_args()
    ray.init(address=args.ray_address, num_cpus=6 if args.smoke_test else None)
    sched = ASHAScheduler()

    tuner = tune.Tuner(
        tune.with_resources(
            TrainTapNet, resources={"cpu": 3, "gpu": int(args.use_gpu)}
        ),
        run_config=air.RunConfig(
            stop={
                "fbeta": 0.95,
                "training_iteration": 3 if args.smoke_test else 20,
            },
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end=True, checkpoint_frequency=3
            ),
        ),
        tune_config=tune.TuneConfig(
            metric="fbeta",
            mode="max",
            scheduler=sched,
            num_samples=1 if args.smoke_test else 20,
        ),
        param_space={
            "args": args,
            "lr": tune.uniform(0.001, 0.1),
            "momentum": tune.uniform(0.1, 0.9),
        },
    )
    results = tuner.fit()

    print("Best config is:", results.get_best_result().config)
