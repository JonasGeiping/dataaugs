"""Defines the main analysis function that can be called to analyze the current state of a model."""

import torch
import torchvision

from .welford import WelfordAccumulation
from .rollouts import perturb2threshold, normalize_direction, get_weights
from .cosine_pairs import compute_cosinesim_blevel, compute_cosinesim
from ..utils import get_log

from ..models.modules import disable_running_stats, enable_running_stats
import copy


def analyze(model, loss_fn, optimizer, augmented_dataloader, stats, setup, cfg):
    """Collect some statistics about the current model.

    This function requires knowledge of cfg.analysis, but also cfg.impl and cfg.data.
    """
    log = get_log(cfg)
    disable_running_stats(model)

    # needs to be reshape for channels_last:
    param_vector = torch.cat([param.cpu().reshape(-1) for param in model.parameters()])

    # record model size for crosschecks:
    num_params, num_buffers = (
        sum([p.numel() for p in model.parameters()]),
        sum([b.numel() for b in model.buffers()]),
    )
    stats["num_parameters"] += [num_params]
    stats["num_buffers"] += [num_buffers]

    if cfg.analysis.measure_param_norm:
        stats["analysis_param_norm"] += [param_vector.norm().item()]  # Saxe

    if cfg.analysis.measure_grad_norm:
        norm_type = cfg.hyp.grad_clip_norm
        try:
            if norm_type == float("inf"):
                stats["analysis_grad_norm"] += [max(p.grad.abs().max() for p in model.parameters())]
            else:
                stats["analysis_grad_norm"] += [
                    torch.norm(torch.stack([torch.norm(p.grad, norm_type) for p in model.parameters()]), norm_type).item()
                ]  # this is the "pytorch" norm-of-norm l2-norm
        except AttributeError:  # sometimes not all gradients have been recorded
            stats["analysis_grad_norm"] += [float("NaN")]

    if cfg.analysis.check_momentum:
        if "momentum" in cfg.hyp.optim and cfg.hyp.optim.momentum > 0:
            grad = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
            momentum = torch.cat([optimizer.state[p]["momentum_buffer"].reshape(-1) for p in model.parameters()])

            stats["analysis_momentum_dist"] += [torch.linalg.norm(grad - momentum).item()]
            stats["analysis_momentum_sim"] += [((grad * momentum).sum() / grad.norm() / momentum.norm()).item()]
        else:
            stats["analysis_momentum_dist"] += [float("NaN")]
            stats["analysis_momentum_sim"] += [float("NaN")]

    # Do we want advanced gradient statistics (mean/variance)?
    if cfg.analysis.analyze_augmented_dataset:
        dataloader = augmented_dataloader
    else:
        dataloader = _get_unaugmented_dataloader(augmented_dataloader, cfg.data)

    if cfg.analysis.compute_gradient_SNR or cfg.analysis.compute_gradient_noise_scale or cfg.analysis.record_gradient_norm_per_batch:
        model.train()
        grads = []
        num_blocks = len(dataloader)

        collector = WelfordAccumulation()

        def collect_gradients(inputs, labels):
            inputs = inputs.to(**setup, non_blocking=cfg.impl.non_blocking)
            labels = labels.to(dtype=torch.long, device=setup["device"], non_blocking=cfg.impl.non_blocking)

            outputs = model(inputs)
            block_loss = loss_fn(outputs, labels) / num_blocks

            grad_list = torch.autograd.grad(block_loss, model.parameters())

            if cfg.analysis.normalize_in_SNR_computations:
                normalize_direction(grad_list, get_weights(model, setup=setup), norm="filter")
            grad_vector = torch.cat([g.detach().cpu().reshape(-1) for g in grad_list])
            collector(grad_vector)
            return grad_vector.norm()

        # Compute gradient informations [this is a limited sample in a DDP distributed setting]
        subblock_counter = 0
        grad_norms = torch.zeros(num_blocks * cfg.analysis.internal_batch_size_chunks, device=setup["device"], dtype=setup["dtype"])
        for block, (inputs, labels) in enumerate(dataloader):
            input_chunks = torch.chunk(inputs, cfg.analysis.internal_batch_size_chunks, dim=0)
            label_chunks = torch.chunk(labels, cfg.analysis.internal_batch_size_chunks, dim=0)
            for input, label in zip(input_chunks, label_chunks):
                grad_norms[subblock_counter] = collect_gradients(input, label)
                subblock_counter += 1

        # Itemization to records separated from computations to improve GPU queuing:
        if cfg.analysis.record_gradient_norm_per_batch:
            for i in range(subblock_counter):
                stats[f"analysis_grad_norm_{i}"] += [grad_norms[i].item()]
        grad_mean, grad_variance, grad_std, grad_norm, squared_norm = collector.finalize()
        # import ipdb; ipdb.set_trace()
        if cfg.analysis.compute_gradient_SNR:
            stats["analysis_grad_mean_mean"] += [grad_mean.mean().item()]  # Saxe
            stats["analysis_grad_mean_norm"] += [grad_mean.norm().item()]  # Saxe
            stats["analysis_grad_std_mean"] += [grad_std.mean().item()]  # Saxe
            stats["analysis_grad_std_norm"] += [grad_std.norm().item()]  # Saxe
            stats["analysis_grad_SNR"] += [stats["analysis_grad_mean_norm"][-1] / (stats["analysis_grad_std_norm"][-1] + 1e-10)]
            log.info(f'Gradient SNR is {stats["analysis_grad_SNR"][-1]}')

        if cfg.analysis.compute_gradient_noise_scale:
            b_local = cfg.data.batch_size // cfg.analysis.internal_batch_size_chunks
            b_full = max(len(dataloader.dataset), cfg.data.size)  # Dataset size might have been artifically inflated
            g_local = squared_norm
            g_full = grad_mean.pow(2).sum()

            candlish_S = 1 / (1 / b_local - 1 / b_full + 1e-10) * (g_local - g_full)
            candlish_G = 1 / (b_full - b_local + 1e-10) * (b_full * g_full - b_local * g_local)
            stats["analysis_grad_noise_scale"] += [(candlish_S / candlish_G).item()]
            log.info(f'Gradient Noise Scale is {stats["analysis_grad_noise_scale"][-1]}')

    if cfg.analysis.compute_flatness:
        try:
            empirical_flatness, counter = perturb2threshold(
                model,
                dataloader,
                torch.nn.CrossEntropyLoss(reduction="sum"),
                setup,
                step_size=cfg.analysis.flatness_step_size,
                threshold=cfg.analysis.flatness_threshold,
                norm=cfg.analysis.flatness_norm,
                ignore="biasbn",
                dryrun=cfg.dryrun,
            )
            stats["analysis_empirical_flatness"] += [empirical_flatness]
            log.info(
                f"Empirical flatness from random directions with threshold {cfg.analysis.flatness_threshold} "
                f'is {stats["analysis_empirical_flatness"][-1]} after {counter} steps.'
            )
        except torch.cuda.OutOfMemoryError:
            log.info("CUDA OUT OF MEMORY IN ANALYSIS FOR empirical flatness")
            stats["analysis_empirical_flatness"] += [float("NaN")]

    try:
        if cfg.analysis.measure_cosinesim_imlevel:
            cossim_stats = compute_cosinesim(model, dataloader, torch.nn.CrossEntropyLoss(reduction="none"), setup, cfg)
            stats["sample_cossim_mean"] += [cossim_stats[0]]
            stats["sample_cossim_std"] += [cossim_stats[1]]
            stats["sample_cossim_pairs"] += [cossim_stats[2]]
            log.info(f'Cosine similarity over {stats["sample_cossim_pairs"][-1]} pairs is {stats["sample_cossim_mean"][-1]}')

        if cfg.analysis.measure_cosinesim_batchlevel:
            cossim_stats = compute_cosinesim_blevel(model, dataloader, torch.nn.CrossEntropyLoss(), setup, cfg)
            stats["batch_cossim_mean"] += [cossim_stats[0]]
            stats["batch_cossim_std"] += [cossim_stats[1]]
            stats["batch_cossim_pairs"] += [cossim_stats[2]]
            log.info(f'Batch level cosine similarity over {stats["batch_cossim_pairs"][-1]} pairs is {stats["batch_cossim_mean"][-1]}')
    except torch.cuda.OutOfMemoryError:
        log.info("CUDA OUT OF MEMORY IN ANALYSIS FOR gradient cosine similarity")
        if cfg.analysis.measure_cosinesim_imlevel:
            stats["sample_cossim_mean"] += [float("NaN")]
            stats["sample_cossim_std"] += [float("NaN")]
            stats["sample_cossim_pairs"] += [float("NaN")]

        if cfg.analysis.measure_cosinesim_batchlevel:
            stats["batch_cossim_mean"] += [float("NaN")]
            stats["batch_cossim_std"] += [float("NaN")]
            stats["batch_cossim_pairs"] += [float("NaN")]

    model.train()
    enable_running_stats(model)

    return stats


def _get_unaugmented_dataloader(dataloader, cfg_data):
    """Only works for datasets from this code-base where all augmentations are in dataset.transform.

    Returns a random-sampling unaugmented dataloader.
    """
    dataset_without_augmentations = copy.deepcopy(dataloader.dataset)
    filtered_transforms = []
    for transform in dataset_without_augmentations.transform.transforms:
        if isinstance(transform, torchvision.transforms.Normalize):
            filtered_transforms.append(transform)
        if isinstance(transform, torchvision.transforms.ToTensor):
            filtered_transforms.append(transform)
    dataset_without_augmentations.transform.transforms = filtered_transforms

    dataloader_without_augmentations = torch.utils.data.DataLoader(
        dataset_without_augmentations,
        batch_size=min(cfg_data.batch_size, len(dataset_without_augmentations)),
        shuffle=True,
        drop_last=True,
        num_workers=min(torch.get_num_threads(), 16) // max(1, torch.cuda.device_count()),
        pin_memory=True,
        persistent_workers=False,
    )
    return dataloader_without_augmentations
