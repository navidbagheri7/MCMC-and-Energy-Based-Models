"""Normalizable Gaussian-base energy models and negative-phase learning.

CD-k, persistent CD, replay and quadrature maximum likelihood share data,
network, update count and validation-only selection. Finite-chain negative
phases are approximate; bounded residual energies make integrability explicit.
"""

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_moons
from torch import nn

from .artifacts import environment, sha256, source_identity, write_json

METHODS = ("cd1", "cd10", "pcd10", "pcd50", "replay20", "quadrature_mle")


class Energy(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    def forward(self, x):
        return 0.5 * (x / 3).square().sum(-1) + 8 * torch.tanh(self.network(x).squeeze(-1) / 8)


def langevin(model, initial, steps, eta, rng):
    x = initial.detach().clone()
    for _ in range(steps):
        x.requires_grad_(True)
        gradient = torch.autograd.grad(model(x).sum(), x)[0]
        noise = torch.randn(x.shape, device=x.device, generator=rng)
        x = (x - eta * gradient + np.sqrt(2 * eta) * noise).detach()
        if not torch.isfinite(x).all():
            raise FloatingPointError("Nonfinite EBM negative chain")
    return x


def grid_geometry(size=81, limit=14, device="cpu"):
    axis = torch.linspace(-limit, limit, size, dtype=torch.float32, device=device)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack([x.flatten(), y.flatten()], 1)
    weights = torch.ones(size, size, device=device, dtype=torch.float64)
    weights[[0, -1]] *= 0.5
    weights[:, [0, -1]] *= 0.5
    weights *= (2 * limit / (size - 1)) ** 2
    return grid, weights.flatten()


def log_partition(model, size=81, limit=14):
    grid, weights = grid_geometry(size, limit, next(model.parameters()).device)
    return torch.logsumexp(-model(grid) + weights.log(), 0)


def dataset(task):
    rng = np.random.default_rng(222000)
    if task == "mixture":
        angle = np.arange(8) * 2 * np.pi / 8
        centers = 3 * np.stack([np.cos(angle), np.sin(angle)], 1)
        data = centers[rng.integers(8, size=10000)] + 0.35 * rng.normal(size=(10000, 2))
    elif task == "moons":
        data, _ = make_moons(n_samples=10000, noise=0.08, random_state=222000)
        data = (data - np.array([0.5, 0.25])) * 2
    else:
        raise ValueError("Unknown EBM task")
    order = rng.permutation(len(data))
    data = data[order].astype(np.float32)
    return data[:8000], data[8000:9000], data[9000:]


def export(model, path):
    np.savez_compressed(
        path, **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    )
    write_json(
        Path(path).with_suffix(".json"),
        dict(schema=1, base_scale=3, residual_bound=8, weights_sha256=sha256(path)),
    )


def load(path):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    if (
        meta.get("schema") != 1
        or meta["base_scale"] != 3
        or meta["residual_bound"] != 8
        or meta["weights_sha256"] != sha256(path)
    ):
        raise ValueError("Unsupported or changed energy model")
    model = Energy().eval()
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays) != set(model.state_dict()):
            raise ValueError("Weight schema mismatch")
        state = {}
        for key, tensor in model.state_dict().items():
            array = arrays[key]
            if (
                array.dtype != np.float32
                or array.shape != tuple(tensor.shape)
                or not np.isfinite(array).all()
            ):
                raise ValueError("Invalid EBM weights")
            state[key] = torch.from_numpy(array.copy())
    model.load_state_dict(state)
    return model


def train(config, output, source_root=Path("assets/source_data")):
    task, method, seed = config["task"], config["method"], config["seed"]
    if method not in METHODS:
        raise ValueError("Unknown negative-phase method")
    name = f"ebm_{task}_{method}_{seed}"
    folder = Path(output) / name
    source = source_identity()
    if (folder / "metrics.json").exists():
        old = json.loads((folder / "run.json").read_text())
        if old["config"] != config or old["source_sha256"] != source:
            raise ValueError("Existing evidence belongs to a different protocol/source")
        assert sha256(source_root / (name + ".npz")) == old["numeric_sha256"]
        print("Verified", name, flush=True)
        return
    if (folder / "run.json").exists():
        raise ValueError("Incomplete run retained; use a new output")
    write_json(
        folder / "run.json",
        dict(status="running", config=config, source_sha256=source, environment=environment()),
    )
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    device = torch.device(config.get("device", "cpu"))
    model = Energy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    data_rng = torch.Generator(device=device).manual_seed(seed + 1000)
    negative_rng = torch.Generator(device=device).manual_seed(seed + 2000)
    train_data, validation, test = dataset(task)
    positives = torch.as_tensor(train_data, device=device)
    val = torch.as_tensor(validation, device=device)
    test_tensor = torch.as_tensor(test, device=device)
    persistent = 3 * torch.randn(1024, 2, device=device, generator=negative_rng)
    best = copy.deepcopy(model.state_dict())
    best_score = np.inf
    best_step = 0
    history = []
    started = time.perf_counter()
    negative_gradient_calls = 0
    integration_grid, integration_weights = grid_geometry(81, 14, device)
    for step in range(config["steps"] + 1):
        if step % 100 == 0 or step == config["steps"]:
            with torch.no_grad():
                logZ = log_partition(model, 161, 14)
                nll = (model(val) + logZ).mean().item()
            if nll < best_score:
                best_score = nll
                best = copy.deepcopy(model.state_dict())
                best_step = step
            history.append(dict(step=step, validation_nll=nll))
            print(name, step, nll, "best", best_step, flush=True)
        if step == config["steps"]:
            break
        selected = torch.randperm(len(positives), device=device, generator=data_rng)[:128]
        real = positives[selected]
        if method == "quadrature_mle":
            negative_loss = torch.logsumexp(-model(integration_grid) + integration_weights.log(), 0)
            loss = model(real).mean() + negative_loss
        else:
            k = int("".join(c for c in method if c.isdigit()))
            if method.startswith("cd"):
                initial = real
            else:
                indices = torch.randperm(len(persistent), device=device, generator=data_rng)[:128]
                initial = persistent[indices].clone()
                if method == "replay20":
                    refresh = torch.rand(len(initial), device=device, generator=negative_rng) < 0.05
                    initial[refresh] = 3 * torch.randn(
                        (int(refresh.sum()), 2), device=device, generator=negative_rng
                    )
            negative = langevin(model, initial, k, 0.02, negative_rng)
            negative_gradient_calls += k
            if not method.startswith("cd"):
                persistent[indices] = negative
            loss = model(real).mean() - model(negative.detach()).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite EBM objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    fitting_seconds = time.perf_counter() - started
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        partition = [
            float(log_partition(model, size, limit))
            for size, limit in [(161, 14), (321, 14), (401, 18)]
        ]
        nll = float((model(test_tensor) + partition[-1]).mean())
        grid, weights = grid_geometry(161, 10, device)
        density = torch.exp(-model(grid) - partition[-1])
    initial = 3 * torch.randn(
        1024, 2, device=device, generator=torch.Generator(device=device).manual_seed(333000)
    )
    # Longer independent negatives expose the fitted stationary geometry; they are
    # approximate ULA draws and do not enter model selection or normalized NLL.
    generated = langevin(
        model, initial, 1000, 0.005, torch.Generator(device=device).manual_seed(444000)
    )
    tail_bound = 2 * np.pi * 9 * np.exp(8) * np.exp(-(18**2) / 18)
    folder.mkdir(parents=True, exist_ok=True)
    export(model, folder / "energy.npz")
    with torch.no_grad():
        torch.testing.assert_close(
            load(folder / "energy.npz")(torch.as_tensor(test)),
            model(test_tensor).cpu(),
            rtol=0,
            atol=0,
        )
    arrays = dict(
        train=train_data,
        validation=validation,
        test=test,
        generated=generated.cpu().numpy(),
        grid=grid.cpu().numpy(),
        density=density.cpu().numpy(),
        history_steps=np.asarray([r["step"] for r in history]),
        history_nll=np.asarray([r["validation_nll"] for r in history]),
        persistent=persistent.cpu().numpy(),
    )
    source_root.mkdir(parents=True, exist_ok=True)
    path = source_root / (name + ".npz")
    np.savez_compressed(path, **arrays)
    write_json(folder / "history.json", history)
    write_json(
        folder / "metrics.json",
        dict(
            config=config,
            selected_step=best_step,
            validation_nll=best_score,
            test_nll=nll,
            log_partition=partition[-1],
            partition_refinements=partition,
            partition_max_change=float(np.ptp(partition)),
            absolute_partition_tail_bound_outside_square18=tail_bound,
            integrability="E(x)=||x||²/18 + bounded residual in [-8,8]; outside-square18 partition contribution <= Gaussian outside-radius18 envelope; quadrature truncation bound is conservative. Finite-grid integration error is estimated by refinement, not rigorously bounded.",
            fitting_seconds=fitting_seconds,
            negative_gradient_evaluations=negative_gradient_calls,
            model_sha256=sha256(folder / "energy.npz"),
            selection="First minimum held-out normalized NLL at 100-update intervals, including initial and final; fixed 161-grid/±14. Test evaluated once at independently refined 401-grid/±18.",
            scope="Owned 2D model study, 8000/1000/1000 fixed independent procedural draws. Equal optimizer updates, unequal negative-phase work. CD/PCD/replay and ULA finite-step bias are retained; generator draws are approximate and do not define NLL. No high-dimensional or image quality claim.",
        ),
    )
    write_json(
        folder / "run.json",
        dict(
            status="complete",
            environment=environment(),
            config=config,
            source_sha256=source,
            numeric_source=str(path),
            numeric_sha256=sha256(path),
        ),
    )
