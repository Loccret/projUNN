import torch
import torch.nn as nn
import numpy as np
import argparse
from pathlib import Path
from torchvision import datasets, transforms
import projunn


def default_data_root():
    repo_data = Path(__file__).resolve().parents[1] / "data"
    if repo_data.exists():
        return repo_data
    return Path("./mnist")


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Exponential Layer MNIST Task")
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=170)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("-lr", "--lr", type=float, default=0.001)
    parser.add_argument("--permute", action="store_true")
    parser.add_argument(
        "--optimizer", type=str, default="RMSProp", choices=["RMSProp", "SGD"]
    )
    parser.add_argument(
        "--projector", type=str, default="projUNND", choices=["projUNND", "projUNNT"]
    )
    parser.add_argument("--rank", type=int, default=1)
    return parser.parse_args(argv)


def configure_reproducibility():
    # Same seed as "Orthogonal Recurrent Neural Networks with Scaled Cayley Transform".
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(5544)
    np.random.seed(5544)


def make_permutation():
    permute = np.random.RandomState(92916)
    return torch.LongTensor(permute.permutation(784))


def make_projector(args):
    def projector(param, update):
        a, b = projunn.utils.LSI_approximation(update, k=args.rank)
        if args.projector == "projUNND":
            update = projunn.utils.projUNN_D(param.data, a, b, project_on=False)
        else:
            update = projunn.utils.projUNN_T(param.data, a, b, project_on=False)
        return update

    return projector


class proj_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(proj_net, self).__init__()
        self.rnn_layer = projunn.layers.OrthogonalRNN(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        hidden = None
        for input in torch.unbind(inputs, dim=1):
            hidden, output = self.rnn_layer(input, hidden)
        output = self.output_layer(hidden)
        return output




def main(argv=None):
    args = parse_args(argv)
    configure_reproducibility()
    permutation = make_permutation()

    # Load data
    kwargs = {"num_workers": 1, "pin_memory": True}
    data_root = default_data_root()
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            data_root, train=True, download=True, transform=transforms.ToTensor()
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            data_root, train=False, download=True, transform=transforms.ToTensor()
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )



    # Model and optimizers
    model = proj_net(1, args.hidden_size, 10).cuda()
    model.train()
    projector = make_projector(args)
    if args.optimizer == "RMSProp":
        optimizer = projunn.optimizers.RMSprop(
            model.parameters(), projector=projector, lr=args.lr
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 60*len(train_loader), 0.2)

    accuracy = AccuracyMeter().cuda()


    for epoch in range(args.epochs):
        accuracy.reset()
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            if args.permute:
                batch_x = batch_x.cuda().view(-1, 784, 1)[:, permutation]
            else:
                batch_x = batch_x.cuda().view(-1, 784, 1)
            predictions = model(batch_x)
            loss = torch.nn.functional.cross_entropy(predictions, batch_y.cuda())
            model.zero_grad()
            loss.backward()
            if args.optimizer == "RMSProp":
                optimizer.step()
                scheduler.step()
            else:
                for param in model.parameters():
                    update = -args.lr * param.grad
                    if hasattr(param, "needs_projection"):
                        update = projector(param, update)
                    param.data.add_(update)
            accuracy.update(predictions, batch_y.cuda())
            W = model.rnn_layer.recurrent_kernel.weight
            print("Unitary?", (W.T @ W - torch.eye(W.size(1), device="cuda")).norm())
            print(
                "Train Epoch: {} ({:.0f}%)\tLoss: {:.6f}\tAccuracy: {:.2f}%".format(
                    epoch,
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    100 * accuracy.compute(),
                )
            )

        accuracy.reset()
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                if args.permute:
                    batch_x = batch_x.cuda().view(-1, 784, 1)[:, permutation]
                else:
                    batch_x = batch_x.cuda().view(-1, 784, 1)
                logits = model(batch_x)
                accuracy.update(logits, batch_y.cuda())
        print()
        print("Test set accuracy: ", accuracy.compute())

        model.train()


if __name__ == "__main__":
    main()
