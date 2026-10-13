import torch
from torchvision import datasets

RAW_DIR = "data"
OUT_PATH = "data/mnist.pt"


def build() -> dict[str, torch.Tensor]:
    train = datasets.MNIST(RAW_DIR, train=True, download=True)
    test = datasets.MNIST(RAW_DIR, train=False, download=True)
    return {
        "x_train": train.data,
        "y_train": train.targets,
        "x_test": test.data,
        "y_test": test.targets,
    }


def load(path: str = OUT_PATH):
    d = torch.load(path)
    return (
        d["x_train"].float() / 255,
        d["y_train"],
        d["x_test"].float() / 255,
        d["y_test"],
    )


def summarize(d: dict[str, torch.Tensor]) -> None:
    balance = torch.bincount(d["y_train"]).tolist()
    print(f"train={tuple(d['x_train'].shape)}  test={tuple(d['x_test'].shape)}")
    print(f"pixel range=[{d['x_train'].min()}, {d['x_train'].max()}]")
    print(f"class balance: {balance}")


def main() -> None:
    d = build()
    summarize(d)
    torch.save(d, OUT_PATH)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
