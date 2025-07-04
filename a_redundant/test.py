import argparse
import logging
from typing import Optional

import torch
from torch import autocast, optim, nn, GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.transforms import v2
import wandb
from tqdm import tqdm
from contextlib import nullcontext

from models import SimpleTransformer
import utils

hyperparameters = {
    "batch_size": 1024,
    "learning_rate": 0.001,
    "epochs": 20,
    "patience": 2,
    "patch_size": 14,  # base MNIST images are 28x28, so patch size of 7 -> 16 patches
    "model_dim": 128,
    "ffn_dim": 64,
    "num_encoders": 3,
    "num_heads": 4,
    "use_pe": True,  # whether to use positional encoding
    "seed": 42,
}

sweep_config = {
    "method": "bayes",  # Can be 'grid', 'random', or 'bayes'
    "metric": {"name": "test_accuracy", "goal": "maximize"},
    "parameters": {
        "batch_size": {"values": [2048]},
        "learning_rate": {"values": [1e-4, 3e-4]},
        "epochs": {"values": [25, 32, 40, 50]},
        "patience": {"values": [2]},
        "patch_size": {"values": [14]},
        "model_dim": {"values": [512]},
        "ffn_dim": {"values": [1024, 2048, 4096]},
        "num_encoders": {"values": [2, 5]},
        "num_heads": {"values": [4, 8, 16]},
        "use_pe": {"values": [True, False]},
    },
}

parser = argparse.ArgumentParser(description="Train simple model")
parser.add_argument("--entity", help="W and B entity", default="mlx-institute")
parser.add_argument("--project", help="W and B project", default="encoder-only")
parser.add_argument("--sweep", help="Run hyperparameter sweep", action="store_true")
parser.add_argument(
    "--no-save",
    help="Don't save model state (or checkpoints)",
    action="store_true",
)
args = parser.parse_args()


def amp_components(device, train=False):
    if device.type == "cuda" and train:
        return autocast(device.type), GradScaler(device.type)
    else:
        # fall-back: no automatic casting, dummy scaler
        return nullcontext(), GradScaler(enabled=False)


def run_batch(
    dataloader,
    model,
    device,
    loss_fn: nn.Module = nn.CrossEntropyLoss(),
    train: bool = False,
    optimizer: Optional[Optimizer] = None,
    desc: str = "",
):
    """
    Runs one pass over `dataloader`.

    If `train` is True, the model is set to training mode and the optimizer is
    stepped. Otherwise the model is evaluated with torch.no_grad().
    Returns (accuracy %, average_loss) for the epoch.
    """
    model.train() if train else model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss, correct = 0.0, 0

    iterator = tqdm(dataloader, desc=desc)
    context = torch.enable_grad() if train else torch.no_grad()
    maybe_autocast, scaler = amp_components(device, train)
    with context:
        for X, y in iterator:
            X, y = X.to(device), y.to(device)
            with maybe_autocast:
                pred = model(X)
                loss = loss_fn(pred, y)

            total_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            if train:
                if optimizer is None:
                    raise ValueError("Optimizer must be provided when train=True")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

    avg_loss = total_loss / num_batches
    accuracy = 100 * correct / size
    return accuracy, avg_loss


def setup_data():
    """Setup and return datasets and dataloaders."""
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    raw_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    stats_dataloader = DataLoader(raw_data, batch_size=len(raw_data.data), shuffle=False)
    images, _ = next(iter(stats_dataloader))

    train_size = int(0.9 * len(raw_data))
    val_size = len(raw_data) - train_size
    generator = torch.Generator().manual_seed(hyperparameters["seed"])
    training_data, val_data = random_split(raw_data, [train_size, val_size], generator)
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    return training_data, val_data, test_data


def run_single_training(config=None):
    """Run a single training session with given config."""
    if config is None:
        config = hyperparameters

    if config["model_dim"] % config["num_heads"] != 0:
        logging.error(
            f"model_dim must be a multiple of num_heads, {config['model_dim']}, {config['num_heads']} not permitted"
        )
        return None

    device = utils.get_device()

    # Setup data
    training_data, val_data, test_data = setup_data()

    pin_memory = device.type == "cuda"
    num_workers = 8 if device.type == "cuda" else 0
    train_dataloader = DataLoader(
        training_data,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    model = SimpleTransformer(
        patch_size=config["patch_size"],
        model_dim=config["model_dim"],
        ffn_dim=config["ffn_dim"],
        num_encoders=config["num_encoders"],
        num_heads=config["num_heads"],
        use_pe=config["use_pe"],
    )
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    wandb.watch(model, log="all", log_freq=100)
    wandb.define_metric("val_accuracy", summary="max")
    wandb.define_metric("val_loss", summary="min")

    logging.info("Starting training...")
    model = run_training(
        model=model,
        train_dl=train_dataloader,
        val_dl=val_dataloader,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        config=config,
    )

    # if we stopped early and have a checkpoint, load it
    if not args.no_save:
        checkpoint = torch.load(utils.SIMPLE_MODEL_FILE)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_correct, test_loss = run_batch(
        dataloader=test_dataloader,
        model=model,
        device=device,
        loss_fn=loss_fn,
        train=False,
        desc="Testing",
    )
    wandb.log({"test_accuracy": test_correct, "test_loss": test_loss})
    logging.info(f"Test accuracy: {test_correct:.2f}%")

    return model


def main():
    utils.setup_logging()
    device = utils.get_device()
    logging.info(f"Using {device} device. Will save? {not args.no_save}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if device.type == "cuda":
        torch.set_float32_matmul_precision("medium")

    if args.sweep:
        run_sweep()
    else:
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            config=hyperparameters,
        )

        run_single_training(hyperparameters)

        if not args.no_save:
            logging.info(f"Saved PyTorch Model State to {utils.SIMPLE_MODEL_FILE}")
            artifact = wandb.Artifact(name="simple_model", type="model")
            artifact.add_file(utils.SIMPLE_MODEL_FILE)
            run.log_artifact(artifact)
        run.finish(0)


def run_training(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    config: dict,
) -> nn.Module:
    best_loss = float("inf")
    epochs_since_best = 0
    for epoch in range(config["epochs"]):
        train_correct, train_loss = run_batch(
            dataloader=train_dl,
            model=model,
            device=device,
            train=True,
            loss_fn=loss_fn,
            optimizer=optimizer,
            desc=f"Training epoch {epoch + 1}",
        )
        val_correct, val_loss = run_batch(
            dataloader=val_dl,
            model=model,
            device=device,
            train=False,
            loss_fn=loss_fn,
            desc=f"Validating epoch {epoch + 1}",
        )
        wandb.log(
            {
                "train_accuracy": train_correct,
                "train_loss": train_loss,
                "val_accuracy": val_correct,
                "val_loss": val_loss,
            },
        )
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_since_best = 0
            if not args.no_save:
                model_dict = {
                    "model_state_dict": model.state_dict(),
                }
                torch.save(model_dict, utils.SIMPLE_MODEL_FILE)
        else:
            epochs_since_best += 1
        if epochs_since_best >= config["patience"]:
            break

    return model


def run_sweep():
    """Run hyperparameter sweep with wandb."""
    print("Starting hyperparameter sweep...")
    print(f"Sweep configuration: {sweep_config}")

    sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)
    print(f"Sweep ID: {sweep_id}")
    print("Run the following command to start agents:")
    print(f"wandb agent {args.entity}/{args.project}/{sweep_id}")

    wandb.agent(
        sweep_id=sweep_id,
        function=sweep_train,
        project=args.project,
        count=40,
    )


def sweep_train():
    """Training function for wandb sweep."""
    # Initialize wandb run with sweep config
    run = wandb.init()
    config = wandb.config

    print("Running training with hyperparameters:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Run training with sweep parameters
    model = run_single_training(config)

    run.finish(0)
    return model


if __name__ == "__main__":
    main()