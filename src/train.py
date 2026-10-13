import json
import os
import platform
import time

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from data import load
from model import make_model

MODEL_PATH = "models/model.pt"
METRICS_PATH = "metrics.json"


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(dim=-1) == y).sum().item()
        total += y.numel()
    return correct / total


def fit(params: dict, data, device: torch.device, log=print):
    x_train, y_train, x_test, y_test = data
    torch.manual_seed(params["seed"])

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=params["batch_size"], shuffle=True
    )
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=1024)

    model = make_model(params["width"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"])
    loss_fn = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, params["epochs"] + 1):
        model.train()
        start = time.perf_counter()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(y)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start

        row = {
            "epoch": epoch,
            "train_loss": round(total_loss / len(x_train), 4),
            "test_acc": round(evaluate(model, test_loader, device), 4),
            "seconds": round(seconds, 2),
        }
        history.append(row)
        log(row)
    return model, history


def main() -> None:
    params = load_params()
    device = pick_device(params["device"])
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"device={name}  params={params}")

    def log(row: dict) -> None:
        print("  ".join(f"{k} {v}" for k, v in row.items()), flush=True)

    model, history = fit(params, load(), device, log=log)

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    last = history[-1]
    metrics = {
        "test_acc": last["test_acc"],
        "train_loss": last["train_loss"],
        "sec_per_epoch": round(sum(r["seconds"] for r in history) / len(history), 2),
        "hardware": name,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "container": os.environ.get("APPTAINER_NAME", "none"),
    }
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
