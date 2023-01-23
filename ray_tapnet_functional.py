# Original Code here:
# https://github.com/pytorch/examples/blob/master/mnist/main.py

import argparse
import os
import shutil
import sys
from math import floor
from pathlib import Path
from shutil import rmtree

from ray import air, tune
from ray.air import session
from ray.tune.schedulers import AsyncHyperBandScheduler
from torchmetrics import ConfusionMatrix
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tapnet import TapNet
from utils import (
    boolean_string,
    euclidean_dist,
    fbeta,
    focal_loss,
    load_custom_ts,
    output_conv_size,
)

# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 512
TEST_SIZE = 256


def train(model, optimizer, model_params, config, device=None):
    args = config.get("args")
    device = device or torch.device("cpu")
    features, labels, idx_train, idx_val, idx_test = model_params
    loss_list = [float(sys.maxsize)]
    test_best_possible, best_so_far = 0.0, sys.maxsize
    for epoch in range(args.epochs):

        model.train()
        optimizer.zero_grad()

        output, proto_dist = model(model_params)
        loss_train = focal_loss(
            output[idx_train],
            labels[idx_train],
        )

        if args.use_metric:
            loss_train = loss_train + args.metric_param * proto_dist

        loss_delta = abs(loss_train.item() - loss_list[-1])
        if loss_delta < args.stop_thres or loss_train.item() > loss_list[-1]:
            break
        else:
            loss_list.append(loss_train.item())

        loss_train.backward()
        optimizer.step()


def validate(model, model_params, config, device=None):
    device = device or torch.device("cpu")
    features, labels, idx_train, idx_val, idx_test = model_params
    model.eval()
    with torch.no_grad():
        output, proto_dist = model(model_params)
        loss_val = focal_loss(
            output[idx_val],
            labels[idx_val],
        )

        return fbeta(output[idx_val], labels[idx_val]), loss_val


def train_tapnet(config):
    args = config.get("args")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    (
        features,
        labels,
        idx_train,
        idx_val,
        idx_test,
        nclass,
    ) = args.custom_ts
    if use_cuda:
        features, labels, idx_train = (
            features.cuda(),
            labels.cuda(),
            idx_train.cuda(),
        )

    # update random permutation parameter
    if args.rp_params[0] < 0:
        dim = features.shape[1]
        args.rp_params = [3, floor(dim / (3 / 2))]
    else:
        dim = features.shape[1]
        args.rp_params[1] = floor(dim / args.rp_params[1])

    args.rp_params = [int(l) for l in args.rp_params]

    # update dilation parameter
    if args.dilation == -1:
        args.dilation = floor(features.shape[2] / 64)

    model_params = (features, labels, idx_train, idx_val, idx_test)
    # model = TapNet().to(device)
    model = TapNet(
        nfeat=features.shape[1],
        len_ts=features.shape[2],
        layers=args.layers,
        nclass=nclass,
        dropout=args.dropout,
        use_lstm=args.use_lstm,
        use_cnn=args.use_cnn,
        filters=args.filters,
        dilation=args.dilation,
        kernels=args.kernels,
        use_metric=args.use_metric,
        use_rp=args.use_rp,
        rp_params=args.rp_params,
        lstm_dim=args.lstm_dim,
    ).to(device)

    # init the optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get("lr", 0.01),
        weight_decay=config.get("wd", 0.9),
    )

    while True:
        train(model, optimizer, model_params, config, device)
        fbeta, loss = validate(model, model_params, config, device)
        if fbeta > model.best_fbeta:
            torch.save(
                model.state_dict(),
                Path(f"{session.get_trial_dir()}/checkpoint.pth"),
            )
            model.best_fbeta = float(fbeta)
        # Set this to run Tune.
        session.report({"fbeta": fbeta, "loss": loss})


def test_best_model(best_result, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    (
        features,
        labels,
        idx_train,
        idx_val,
        idx_test,
        nclass,
    ) = args.custom_ts
    if use_cuda:
        features, labels, idx_train = (
            features.cuda(),
            labels.cuda(),
            idx_train.cuda(),
        )

    # update random permutation parameter
    if args.rp_params[0] < 0:
        dim = features.shape[1]
        args.rp_params = [3, floor(dim / (3 / 2))]
    else:
        dim = features.shape[1]
        args.rp_params[1] = floor(dim / args.rp_params[1])

    args.rp_params = [int(l) for l in args.rp_params]

    # update dilation parameter
    if args.dilation == -1:
        args.dilation = floor(features.shape[2] / 64)

    model_params = (features, labels, idx_train, idx_val, idx_test)
    # model = TapNet().to(device)
    model = TapNet(
        nfeat=features.shape[1],
        len_ts=features.shape[2],
        layers=args.layers,
        nclass=nclass,
        dropout=args.dropout,
        use_lstm=args.use_lstm,
        use_cnn=args.use_cnn,
        filters=args.filters,
        dilation=args.dilation,
        kernels=args.kernels,
        use_metric=args.use_metric,
        use_rp=args.use_rp,
        rp_params=args.rp_params,
        lstm_dim=args.lstm_dim,
    ).to(device)

    checkpoint_path = os.path.join(best_result.log_dir, "checkpoint.pth")

    model_state = torch.load(checkpoint_path)
    model.load_state_dict(model_state)

    with torch.no_grad():
        output, proto_dist = model(model_params)
        y_true = labels[idx_test]
        y_pred = torch.argmax(output[idx_test], dim=1)
        confmat = ConfusionMatrix(
            task="binary" if nclass == 2 else "multilabel", num_labels=nclass
        )
        cm = confmat(y_true, y_pred)
        fbeta_test = fbeta(output[idx_val], labels[idx_val])
        print(f"CM: {cm}, fbeta: {fbeta_test}")


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch TapNet Tuning")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument(
        "--data_path", type=str, default="./data/", help="Data path."
    )

    parser.add_argument(
        "--local-dir",
        # required=True,
        default="./results",
        action="store",
        dest="local_dir",
        help="name of the log folder",
    )

    # Ray parameters
    parser.add_argument(
        "--local-debug",
        action="store_true",
        help="Enable single threaded debugging.",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Enables GPU training",
    )
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing."
    )
    parser.add_argument(
        "--ray-address",
        help="Address of Ray cluster for seamless distributed execution.",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        required=False,
        help="The address of server to connect to if using Ray Client.",
    )

    # Training parameter settings
    parser.add_argument(
        "--epochs", type=int, default=3000, help="Number of epochs to train."
    )
    parser.add_argument(
        "--stop_thres",
        type=float,
        default=1e-9,
        help="The stop threshold for the training error. If the difference between training losses "
        "between epochs are less than the threshold, the training will be stopped. Default:1e-9",
    )

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
    return parser.parse_known_args()


if __name__ == "__main__":
    args, _ = parse_args()
    args.name = "tapnet"
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

        

    args.sparse = True
    args.layers = [int(l) for l in args.layers.split(",")]
    args.kernels = [int(l) for l in args.kernels.split(",")]
    args.filters = [int(l) for l in args.filters.split(",")]
    args.rp_params = [float(l) for l in args.rp_params.split(",")]

    # load custom_ts once instead for every model
    args.custom_ts = load_custom_ts(path=args.data_path)

    if not os.path.exists(args.local_dir):
        os.makedirs(args.local_dir)

    if args.server_address:
        ray.init(f"ray://{args.server_address}")
    elif args.ray_address:
        ray.init(address=args.ray_address)
    elif args.local_debug:
        ray.init(local_mode=True)  # single threaded debugging
    else:
        ray.init(num_cpus=2 if args.smoke_test else None)

    # for early stopping
    sched = AsyncHyperBandScheduler()

    resources_per_trial = {
        "cpu": float(2),
        "gpu": float(args.cuda),
    }

    try:
        tuner = tune.Tuner(
            tune.with_resources(train_tapnet, resources=resources_per_trial),
            tune_config=tune.TuneConfig(
                metric="fbeta",
                mode="max",
                scheduler=sched,
                num_samples=5 if args.smoke_test else 50,
            ),
            run_config=air.RunConfig(
                name=args.name,
                stop={
                    "fbeta": 0.80,
                    "training_iteration": 50 if args.smoke_test else 500,
                },
                local_dir=args.local_dir,
                checkpoint_config=air.CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_score_attribute="fbeta",
                ),
            ),
            param_space={
                "args": args,
                "lr": tune.loguniform(1e-4, 1e-2),
                "wd": tune.uniform(0.5, 0.9),
            },
        )
        results = tuner.fit()

        best_result = results.get_best_result("fbeta", "max")

        if best_result.metrics is not None:
            print("Best trial config: {}".format(best_result.config))
            print(
                "Best trial final validation loss: {}".format(
                    best_result.metrics["loss"]
                )
            )
            print(
                "Best trial final validation fbeta: {}".format(
                    best_result.metrics["fbeta"]
                )
            )

            test_best_model(best_result, args)

    except Exception as e:
        print(e)
    finally:
        # delete trial folder after execution (to save space)
        trial_folder = Path(f"{args.local_dir}/{args.name}")
        if trial_folder.exists():
            shutil.rmtree(trial_folder)
            print(f"deleted {trial_folder}")
