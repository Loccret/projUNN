"""
File: mnist_experiments.py
Created Date: Wed Mar 09 2022
Author: Randall Balestriero
-----
Last Modified: Wed Mar 09 2022 10:07:14 PM
Modified By: Randall Balestriero
-----
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import torch
import torchvision
from torchvision import transforms
import argparse
import projunn
import numpy as np
import os
from pathlib import Path


def default_data_root():
    repo_data = Path(__file__).resolve().parents[1] / "data"
    if repo_data.exists():
        return repo_data
    return Path("./")


class AccuracyMeter:
    def __init__(self, device=None):
        self.device = device
        self.reset()

    def cuda(self):
        self.device = "cuda"
        return self

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, prediction, labels):
        predicted = prediction.argmax(dim=1)
        self.correct += int((predicted == labels).sum().item())
        self.total += int(labels.numel())

    def __call__(self, prediction, labels):
        self.update(prediction, labels)

    def compute(self):
        value = 0.0 if self.total == 0 else self.correct / self.total
        return torch.tensor(value, device=self.device)


def load_data(name, batch_size, data_root):

    if "CIFAR" in name:
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_tfms = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*stats, inplace=True),
            ]
        )
        valid_tfms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(*stats)]
        )
    else:
        train_tfms = transforms.ToTensor()
        valid_tfms = transforms.ToTensor()
    train = torchvision.datasets.__dict__[name](
        data_root, download=True, train=True, transform=train_tfms
    )
    test = torchvision.datasets.__dict__[name](
        data_root, download=True, train=False, transform=valid_tfms
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False, drop_last=True
    )

    if name == "MNIST":
        epochs = 5
        channels = 1
        length = 28
    else:
        epochs = 90
        channels = 3
        length = 32

    return train_loader, test_loader, epochs, channels, length


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-lr_divider", type=float, default=10.0)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("-bs", type=int, default=256)
    parser.add_argument(
        "--optimizer", type=str, default="RMSProp", choices=["RMSProp", "SGD"]
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        choices=["MNIST", "CIFAR10", "CIFAR100"],
    )
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument(
        "--projector", type=str, default="projUNND", choices=["projUNND", "projUNNT"]
    )
    parser.add_argument("--unitary", action="store_true")
    return parser.parse_args(argv)


def make_projector(args):
    def projector(param, update):
        a, b = projunn.utils.LSI_approximation(update / args.lr_divider, k=args.rank)
        if args.projector == "projUNND":
            update = projunn.utils.projUNN_D(param.data, a, b, project_on=False)
        else:
            update = projunn.utils.projUNN_T(param.data, a, b, project_on=False)
        return update

    return projector


def main(argv=None):
    args = parse_args(argv)
    train_loader, test_loader, epochs, channels, length = load_data(
        args.dataset, args.bs, default_data_root()
    )
    model = projunn.models.ResNet9(
        channels, num_classes=10, image_length=length, unitary=args.unitary
    ).cuda()
    model.train()

    projector = make_projector(args)
    if args.optimizer == "RMSProp":
        optimizer = projunn.optimizers.RMSprop(
            model.parameters(), projector=projector, lr=args.lr
        )

    accuracy = AccuracyMeter().cuda()

    if args.gamma:
        regularizer = projunn.utils.OrthoRegularizer(model, [3, 3])

    for epoch in range(epochs):
        accuracy.reset()
        for iter, (images, labels) in enumerate(train_loader):
            prediction = model(images.cuda())
            if args.gamma:
                loss = (
                    torch.nn.functional.cross_entropy(prediction, labels.cuda())
                    + regularizer.regularize() * args.gamma
                )
            else:
                loss = torch.nn.functional.cross_entropy(prediction, labels.cuda())

            model.zero_grad()
            loss.backward()
            if args.optimizer == "RMSProp":
                optimizer.step()
            else:
                for param in model.parameters():
                    update = -args.lr * param.grad
                    if hasattr(param, "needs_projection"):
                        update = projector(param, update)
                    param.data.add_(update)
            with torch.no_grad():
                accuracy(prediction, labels.cuda())
                if iter % 10 == 0:
                    w = model.conv1[0].weight
                    diff = w @ torch.conj(w.permute(0, 2, 1)) - torch.eye(
                        w.shape[1], device="cuda"
                    )
                    print(
                        f"Current loss is: {loss.item()}, and far from orthogonal? {diff.norm()}"
                    )

        print("epoch train accuracy", accuracy.compute().item())
        accuracy.reset()
        model.eval()
        with torch.no_grad():
            for iter, (images, labels) in enumerate(test_loader):
                prediction = model(images.cuda())
                accuracy(prediction, labels.cuda())
        print("epoch test accuracy", accuracy.compute().item())
        model.train()


if __name__ == "__main__":
    main()
