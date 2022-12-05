# Original Code here:
# https://github.com/pytorch/examples/blob/master/mnist/main.py

from ray import air, tune
from ray.air import session
from ray.tune.schedulers import AsyncHyperBandScheduler
from utils import (
    output_conv_size,
    euclidean_dist,
    focal_loss,
    fbeta,
    load_custom_ts,
    boolean_string,
)
import argparse
import ray
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import floor

# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 512
TEST_SIZE = 256


class TapNet(nn.Module):
    def __init__(
        self,
        nfeat,
        len_ts,
        nclass,
        dropout,
        filters,
        kernels,
        dilation,
        layers,
        use_rp,
        rp_params,
        use_att=True,
        use_metric=False,
        use_lstm=False,
        use_cnn=True,
        lstm_dim=128,
    ):
        super(TapNet, self).__init__()

        self.nclass = nclass
        self.dropout = dropout
        self.use_metric = use_metric
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn

        # parameters for random projection
        self.use_rp = use_rp
        self.rp_group, self.rp_dim = rp_params

        if True:
            # LSTM
            self.channel = nfeat
            self.ts_length = len_ts

            self.lstm_dim = lstm_dim
            self.lstm = nn.LSTM(self.ts_length, self.lstm_dim)

            paddings = [0, 0, 0]
            if self.use_rp:
                self.conv_1_models = nn.ModuleList()
                self.idx = []
                for i in range(self.rp_group):
                    self.conv_1_models.append(
                        nn.Conv1d(
                            self.rp_dim,
                            filters[0],
                            kernel_size=kernels[0],
                            dilation=dilation,
                            stride=1,
                            padding=paddings[0],
                        )
                    )
                    self.idx.append(
                        # np.random.permutation(nfeat)[0 : self.rp_dim]
                        torch.randperm(nfeat)[0 : self.rp_dim]
                    )
            else:
                self.conv_1 = nn.Conv1d(
                    self.channel,
                    filters[0],
                    kernel_size=kernels[0],
                    dilation=dilation,
                    stride=1,
                    padding=paddings[0],
                )

            self.conv_bn_1 = nn.BatchNorm1d(filters[0])

            self.conv_2 = nn.Conv1d(
                filters[0],
                filters[1],
                kernel_size=kernels[1],
                stride=1,
                padding=paddings[1],
            )

            self.conv_bn_2 = nn.BatchNorm1d(filters[1])

            self.conv_3 = nn.Conv1d(
                filters[1],
                filters[2],
                kernel_size=kernels[2],
                stride=1,
                padding=paddings[2],
            )

            self.conv_bn_3 = nn.BatchNorm1d(filters[2])

            # compute the size of input for fully connected layers
            fc_input = 0
            if self.use_cnn:
                conv_size = len_ts
                for i in range(len(filters)):
                    conv_size = output_conv_size(
                        conv_size, kernels[i], 1, paddings[i]
                    )
                fc_input += conv_size
                # * filters[-1]
            if self.use_lstm:
                fc_input += conv_size * self.lstm_dim

            if self.use_rp:
                fc_input = self.rp_group * filters[2] + self.lstm_dim

        # Representation mapping function
        layers = [fc_input] + layers
        print("Layers", layers)
        self.mapping = nn.Sequential()
        for i in range(len(layers) - 2):
            self.mapping.add_module(
                "fc_" + str(i), nn.Linear(layers[i], layers[i + 1])
            )
            self.mapping.add_module(
                "bn_" + str(i), nn.BatchNorm1d(layers[i + 1])
            )
            self.mapping.add_module("relu_" + str(i), nn.LeakyReLU())

        # add last layer
        self.mapping.add_module(
            "fc_" + str(len(layers) - 2), nn.Linear(layers[-2], layers[-1])
        )
        if len(layers) == 2:  # if only one layer, add batch normalization
            self.mapping.add_module(
                "bn_" + str(len(layers) - 2), nn.BatchNorm1d(layers[-1])
            )

        # Attention
        att_dim, semi_att_dim = 128, 128
        self.use_att = use_att
        if self.use_att:
            self.att_models = nn.ModuleList()
            for _ in range(nclass):

                att_model = nn.Sequential(
                    nn.Linear(layers[-1], att_dim),
                    nn.Tanh(),
                    nn.Linear(att_dim, 1),
                )
                self.att_models.append(att_model)

    def forward(self, input):

        (
            x,
            labels,
            idx_train,
            idx_val,
            idx_test,
        ) = input  # x is N * L, where L is the time-series feature dimension

        N = x.size(0)

        # LSTM
        if self.use_lstm:
            x_lstm = self.lstm(x)[0]
            x_lstm = x_lstm.mean(1)
            x_lstm = x_lstm.view(N, -1)

        if self.use_cnn:
            # Covolutional Network
            # input ts: # N * C * L
            if self.use_rp:
                for i in range(len(self.conv_1_models)):
                    # x_conv = x
                    x_conv = self.conv_1_models[i](x[:, self.idx[i], :])
                    x_conv = self.conv_bn_1(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = self.conv_2(x_conv)
                    x_conv = self.conv_bn_2(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = self.conv_3(x_conv)
                    x_conv = self.conv_bn_3(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = torch.mean(x_conv, 2)

                    if i == 0:
                        x_conv_sum = x_conv
                    else:
                        x_conv_sum = torch.cat([x_conv_sum, x_conv], dim=1)

                x_conv = x_conv_sum
            else:
                x_conv = x
                x_conv = self.conv_1(x_conv)  # N * C * L
                x_conv = self.conv_bn_1(x_conv)
                x_conv = F.leaky_relu(x_conv)

                x_conv = self.conv_2(x_conv)
                x_conv = self.conv_bn_2(x_conv)
                x_conv = F.leaky_relu(x_conv)

                x_conv = self.conv_3(x_conv)
                x_conv = self.conv_bn_3(x_conv)
                x_conv = F.leaky_relu(x_conv)

                x_conv = x_conv.view(N, -1)

        if self.use_lstm and self.use_cnn:
            x = torch.cat([x_conv, x_lstm], dim=1)
        elif self.use_lstm:
            x = x_lstm
        elif self.use_cnn:
            x = x_conv

        # linear mapping to low-dimensional space
        x = self.mapping(x)

        # generate the class protocal with dimension C * D (nclass * dim)
        proto_list = []
        for i in range(self.nclass):
            idx = (labels[idx_train].squeeze() == i).nonzero().squeeze(1)
            if self.use_att:
                A = self.att_models[i](x[idx_train][idx])  # N_k * 1
                A = torch.transpose(A, 1, 0)  # 1 * N_k
                A = F.softmax(A, dim=1)  # softmax over N_k

                class_repr = torch.mm(A, x[idx_train][idx])  # 1 * L
                class_repr = torch.transpose(class_repr, 1, 0)  # L * 1
            else:  # if do not use attention, simply use the mean of training samples with the same labels.
                class_repr = x[idx_train][idx].mean(0)  # L * 1
            proto_list.append(class_repr.view(1, -1))
        x_proto = torch.cat(proto_list, dim=0)

        # prototype distance
        proto_dists = euclidean_dist(x_proto, x_proto)
        proto_dists = torch.exp(-0.5 * proto_dists)
        num_proto_pairs = int(self.nclass * (self.nclass - 1) / 2)
        proto_dist = torch.sum(proto_dists) / num_proto_pairs

        dists = euclidean_dist(x, x_proto)

        # dump_embedding(x_proto, x, labels)
        return torch.exp(-0.5 * dists), proto_dist


def train(model, optimizer, model_params, config, device=None):
    args = config.get("args")
    device = device or torch.device("cpu")
    features, labels, idx_train, idx_val, idx_test = model_params
    loss_list = [sys.maxsize]
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


def test(model, model_params, config, device=None):
    device = device or torch.device("cpu")
    features, labels, idx_train, idx_val, idx_test = model_params
    model.eval()
    with torch.no_grad():
        output, proto_dist = model(model_params)
        loss_test = focal_loss(
            output[idx_val],
            labels[idx_val],
        )

        return fbeta(output[idx_val], labels[idx_val]), loss_test


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
    ) = load_custom_ts()
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
        fbeta, loss = test(model, model_params, config, device)
        # Set this to run Tune.
        session.report({"fbeta": fbeta, "loss": loss})

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch TapNet Tuning")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Ray parameters
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
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    args.sparse = True
    args.layers = [int(l) for l in args.layers.split(",")]
    args.kernels = [int(l) for l in args.kernels.split(",")]
    args.filters = [int(l) for l in args.filters.split(",")]
    args.rp_params = [float(l) for l in args.rp_params.split(",")]

    if args.server_address:
        ray.init(f"ray://{args.server_address}")
    elif args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init(num_cpus=2 if args.smoke_test else None)

    # for early stopping
    sched = AsyncHyperBandScheduler()

    resources_per_trial = {
        "cpu": 2,
        "gpu": int(args.cuda),
    }  # set this for GPUs
    tuner = tune.Tuner(
        tune.with_resources(train_tapnet, resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            metric="fbeta",
            mode="max",
            scheduler=sched,
            num_samples=1 if args.smoke_test else 50,
        ),
        run_config=air.RunConfig(
            name="exp",
            stop={
                "fbeta": 0.98,
                "training_iteration": 5 if args.smoke_test else 100,
            },
        ),
        param_space={
            "args": args,
            "lr": tune.loguniform(1e-4, 1e-2),
            "wd": tune.uniform(0.5, 0.9),
        },
    )
    results = tuner.fit()

    print("Best config is:", results.get_best_result().config)
