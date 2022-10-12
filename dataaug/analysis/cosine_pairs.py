"""Evaluation cosine similarity between image pairs or batches."""

import torch
import numpy as np


def flatten_unit(w1):
    """Flatten and normalize vectors"""
    res_list = []

    for w in w1:
        res_list.append(w.reshape(-1))
    res = torch.concat(res_list)
    res = res / torch.norm(res)
    return res


def compute_grad(sample, label, model, loss_fn):
    """Compute sample gradients."""
    sample = sample.unsqueeze(0)  # prepend batch dimension for processing
    label = label.unsqueeze(0)

    loss = loss_fn(model(sample), label)
    g = torch.autograd.grad(loss, model.parameters())
    return flatten_unit(g)


def compute_cosinesim(model, dataloader, loss_fn, setup, cfg=None):
    """Compute per-sample cosine similarities (in eval mode)."""
    model.eval()
    accum_cosinesims = []
    num_pairs = max(int(cfg.analysis.cossim_numpairs), len(dataloader)) // len(dataloader)
    for block, (inputs, labels) in enumerate(dataloader):

        inputs = inputs.to(**setup, non_blocking=cfg.impl.non_blocking)
        labels = labels.to(dtype=torch.long, device=setup["device"], non_blocking=cfg.impl.non_blocking)
        count = 0
        n = inputs.shape[0]
        while count < num_pairs:
            count += 1
            pair = np.random.choice(n, 2)
            cs = torch.dot(
                compute_grad(inputs[pair[0]], labels[pair[0]], model, loss_fn),
                compute_grad(inputs[pair[1]], labels[pair[1]], model, loss_fn),
            )
            accum_cosinesims.append(cs.item())
            if cfg.dryrun:
                break
    mean, std, total_count = np.mean(accum_cosinesims), np.std(accum_cosinesims), len(accum_cosinesims)
    model.train()
    return mean, std, total_count


def compute_grad_blevel(inputs, labels, model, loss_fn, cfg, setup):
    """Batched Gradients"""

    inputs = inputs.to(**setup, non_blocking=cfg.impl.non_blocking)
    labels = labels.to(dtype=torch.long, device=setup["device"], non_blocking=cfg.impl.non_blocking)

    loss = loss_fn(model(inputs), labels)
    g = torch.autograd.grad(loss, model.parameters())
    return flatten_unit(g)


def compute_cosinesim_blevel(model, dataloader, loss_fn, setup, cfg=None):
    """Compute per-batch cosine similarities."""
    model.train()
    n_pairs = cfg.analysis.cossim_numpairs

    accum_cosinesims = []
    count = 0
    dataloader_iterator = iter(dataloader)

    cs1 = compute_grad_blevel(*next(dataloader_iterator), model, loss_fn, cfg, setup)
    try:
        inputs, labels = next(dataloader_iterator)
    except StopIteration:
        dataloader_iterator = iter(dataloader)
        inputs, labels = next(dataloader_iterator)
    cs2 = compute_grad_blevel(inputs, labels, model, loss_fn, cfg, setup)
    while count < n_pairs:
        count += 3

        try:
            inputs, labels = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(dataloader)
            inputs, labels = next(dataloader_iterator)

        cs3 = compute_grad_blevel(inputs, labels, model, loss_fn, cfg, setup)
        accum_cosinesims.append(torch.dot(cs1, cs2).item())
        accum_cosinesims.append(torch.dot(cs1, cs3).item())
        accum_cosinesims.append(torch.dot(cs3, cs2).item())
        cs1 = cs1.clone()
        cs2 = cs3.clone()

        if cfg.dryrun:
            break
    mean, std, total_count = np.mean(accum_cosinesims), np.std(accum_cosinesims), len(accum_cosinesims)

    return mean, std, total_count
