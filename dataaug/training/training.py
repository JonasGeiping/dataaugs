"""Main training routine."""
import torch

import time
from collections import defaultdict

import os

from .optimizers import optim_interface
from ..utils import get_log
from .utils import _save_to_checkpoint, _load_from_checkpoint
from ..models.modules import LabelSmoothCrossEntropyLoss, MaxupLoss, IncorrectCrossEntropyLoss, GradRegularizer
from ..analysis import analyze


def train_sgd(model, trainloader, validloaders, setup, cfg):
    """Simplified script for sanity checks."""
    log = get_log(cfg)
    model.train()
    optimizer, scheduler = optim_interface(model, cfg.hyp)
    stats = defaultdict(list)

    class Counter:
        step: int = 0
        epoch: int = 0

    # Optionally: Load checkpoint:
    if cfg.impl.checkpoint.name is not None:
        file = os.path.join(cfg.original_cwd, "checkpoints", cfg.impl.checkpoint.name)
        _load_from_checkpoint(model, optimizer, scheduler, None, Counter, cfg.hyp.steps, device=setup["device"], file=file)

    num_blocks = len(trainloader)
    num_machines = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    loss_fn = get_loss_fn(cfg.hyp, cfg.data.batch_size)
    gradreg = GradRegularizer(model, optimizer, loss_fn, **cfg.hyp.grad_reg)

    def _compute_batched_gradient(inputs, labels):
        outputs = model(inputs)
        block_loss = loss_fn(outputs, labels)  # scaling happens below
        block_correct_preds = (outputs.argmax(dim=-1) == labels).float().sum()

        grads = torch.autograd.grad(block_loss, model.parameters(), create_graph=False)
        return grads, block_loss.detach(), block_correct_preds.detach()

    @torch.no_grad()
    def _record_stats(stats, pre_grads, step_loss, step_preds, datapoints, train_time):
        # Compute full loss:
        param_norm = sum([p.detach().pow(2).sum() for p in model.parameters()])
        full_loss = step_loss / num_blocks + 0.5 * getattr(cfg.hyp.optim, "weight_decay", 0.0) * param_norm
        if torch.distributed.is_initialized():
            package = torch.stack([step_loss, step_preds, full_loss])
            # Add loss terms from all machines:
            torch.distributed.reduce(package, dst=0, async_op=False)
            step_loss, step_preds, full_loss = package

        stats["train_loss"] += [step_loss.item() / num_machines]
        stats["train_acc"] += [step_preds.item() / datapoints / num_machines]
        stats["train_time"] += [time.time() - train_time]
        stats["param_norm"] += [param_norm.item()]
        stats["full_loss"] += [full_loss.item() / num_machines]

    # ## MAIN LOOP is controlled here ## ###################
    step_loss, step_preds, datapoints, seen_steps = 0.0, 0.0, 0, 0
    average_grads = [torch.zeros_like(p) for p in model.parameters()]

    model.train()
    train_time = time.time()
    while Counter.step < cfg.hyp.steps:

        trainloader.sampler.set_epoch(Counter.epoch)
        for block, (inputs, labels) in enumerate(trainloader):

            # Reset gradients:
            torch._foreach_zero_(average_grads)

            # Define smaller data chunks (this is optional and might be just one chunk)
            chunks_in_block = max(labels.shape[0] // cfg.hyp.sub_batch, 1)
            input_chunks = torch.chunk(inputs, chunks_in_block, dim=0)
            label_chunks = torch.chunk(labels, chunks_in_block, dim=0)

            # Gradient evaluation part:
            datapoints += labels.shape[0]
            block_loss, block_correct_preds = 0.0, 0.0

            for idx, (input_chunk, label_chunk) in enumerate(zip(input_chunks, label_chunks)):
                input_chunk = input_chunk.to(**setup, non_blocking=cfg.impl.non_blocking)
                label_chunk = label_chunk.to(dtype=torch.long, device=setup["device"], non_blocking=cfg.impl.non_blocking)
                # Gradient Accumulation
                grads, chunk_loss, chunk_correct_preds = _compute_batched_gradient(input_chunk, label_chunk)
                grads = gradreg(grads, input_chunk, label_chunk, None)

                torch._foreach_sub_(grads, average_grads)
                torch._foreach_add_(average_grads, grads, alpha=1 / (num_machines * chunks_in_block))

                block_loss += chunk_loss / chunks_in_block
                block_correct_preds += chunk_correct_preds

            step_loss += block_loss.detach()
            step_preds += block_correct_preds.detach()
            seen_steps += 1

            # Return to usual format
            for param, grad in zip(model.parameters(), average_grads):
                param.grad = grad

            if cfg.hyp.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.hyp.grad_clip, norm_type=2.0)

            optimizer.step()
            scheduler.step()  # schedulers count in steps!

            # Validate
            if Counter.step % cfg.impl.validate_every_nth_step == 0 or Counter.step == cfg.hyp.steps or cfg.dryrun:
                evaluate_per_class(model, validloaders, stats, setup, cfg.impl, cfg.hyp, dryrun=cfg.dryrun)

            if Counter.step % cfg.impl.print_every_nth_step == 0 or Counter.step == cfg.hyp.steps:
                _record_stats(stats, None, step_loss / seen_steps, step_preds, datapoints, train_time)
                step_loss, step_preds, datapoints, seen_steps = 0.0, 0.0, 0, 0
                train_time = time.time()
                # Print log
                log.info(status_message(optimizer, stats, Counter.step))

            # Save internal checkpoints from rank 0 [Separate from model dict saves]
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                if cfg.impl.checkpoint.name is not None:
                    if (Counter.step) % cfg.impl.checkpoint.save_every_nth_step == 0 or Counter.step == cfg.hyp.steps:
                        file = os.path.join(cfg.original_cwd, "checkpoints", cfg.impl.checkpoint.name)
                        _save_to_checkpoint(model, optimizer, scheduler, None, Counter, file=file)

            # Run optional analysis stuff
            if cfg.analysis.type is not None:
                if ((Counter.step + 1) % cfg.analysis.check_every_nth_step == 0) or (Counter.step == cfg.hyp.steps) or cfg.dryrun:
                    analyze(model, loss_fn, optimizer, trainloader, stats, setup, cfg)

            Counter.step += 1
            if cfg.dryrun or (Counter.step == cfg.hyp.steps):  # break out of last epoch when step limit is reached.
                break

        Counter.epoch += 1
        # Early stopping if loss is not finite
        if not torch.as_tensor(stats["train_loss"][-1]).isfinite():
            log.info("Terminating iterations due to divergence of loss...")
            break

        if cfg.dryrun:
            break

    return stats


def evaluate_per_class(model, dataloaders, stats, setup, cfg_impl, cfg_hyp, dryrun=False, eval_per_class_acc=False):
    """Validation. In a distributed setting this operation is replicated on all machines and work is not shared.

    Computes per-class validation accuracy in addition to average validation acc.
    """
    loss_fn = torch.nn.CrossEntropyLoss()
    model.eval()
    if cfg_impl.setup.dist:
        # Synchronize statistics across machines if any exist:
        # This block cannot be inside inference_mode
        if len(list(model.buffers())) > 0:
            concat_buf = torch.cat([b.data.reshape(-1) for b in model.buffers()])
            torch.distributed.all_reduce(concat_buf, async_op=False)
            pointer = 0
            for buffer in model.buffers():
                num_values = buffer.numel()
                buffer.data = concat_buf[pointer : pointer + num_values].view_as(buffer) / cfg_impl.setup.world_size
                pointer += num_values

    if stats is None:
        stats = defaultdict(list)

    with torch.inference_mode():
        for name, dataloader in dataloaders.items():
            step_loss, datapoints = 0.0, 0
            num_classes = len(dataloader.dataset.classes)
            correct_preds_per_class, data_point_per_class = [0] * num_classes, [0] * num_classes

            # Iterate over all blocks in the validation dataset
            for block, (inputs, labels) in enumerate(dataloader):
                datapoints += labels.shape[0]
                inputs = inputs.to(**setup, non_blocking=cfg_impl.non_blocking)
                labels = labels.to(dtype=torch.long, device=setup["device"], non_blocking=cfg_impl.non_blocking)

                if cfg_hyp.test_time_flips:
                    outputs = (model(inputs) + model(torch.flip(inputs, [3]))) / 2
                else:
                    outputs = model(inputs)
                block_loss = loss_fn(outputs, labels)
                step_loss += block_loss.item() * labels.shape[0]

                predictions = outputs.argmax(dim=-1)
                for cls_idx in range(num_classes):
                    # Could probably vectorize this...
                    correct_preds_per_class[cls_idx] += (predictions[labels == cls_idx] == cls_idx).sum()
                    data_point_per_class[cls_idx] += (labels == cls_idx).sum().item()

                if dryrun:
                    break

            stats[f"{name}_valid_loss"] += [step_loss / datapoints]
            stats[f"{name}_valid_acc"] += [sum([p.item() for p in correct_preds_per_class]) / datapoints]
            if eval_per_class_acc:
                for cls_idx in range(num_classes):
                    # Stay in torch for Nan propagation
                    cls_acc = correct_preds_per_class[cls_idx] / data_point_per_class[cls_idx]
                    stats[f"{name}_valid_acc_cls{cls_idx}"] += [cls_acc.item()]

    model.train()
    return stats


def get_loss_fn(cfg_hyp, batch_size):
    if cfg_hyp.label_smoothing not in [None, ""]:
        if cfg_hyp.loss_modification is None:
            loss_fn = torch.jit.script(
                LabelSmoothCrossEntropyLoss(smoothing=cfg_hyp.label_smoothing, loss_modification=cfg_hyp.loss_modification)
            )
        elif cfg_hyp.loss_modification == "incorrect-xent":
            loss_fn = torch.jit.script(IncorrectCrossEntropyLoss(smoothing=cfg_hyp.label_smoothing))
        else:
            raise ValueError("Loss modification not implemented in conjunction with label smoothing.")
    else:
        if cfg_hyp.loss_modification is None:
            loss_fn = torch.nn.CrossEntropyLoss()
        elif cfg_hyp.loss_modification == "incorrect-xent":
            loss_fn = torch.jit.script(IncorrectCrossEntropyLoss(smoothing=0.0))
        elif cfg_hyp.loss_modification == "batch-maxup":
            loss_fn = torch.jit.script(MaxupLoss(ntrials=batch_size))
        elif "maxup" in cfg_hyp.loss_modification:
            loss_fn = torch.jit.script(MaxupLoss(ntrials=int(cfg_hyp.loss_modification.split("maxup-")[1])))
        else:
            raise ValueError(f"Invalid loss modification {cfg_hyp.loss_modification}.")

    return loss_fn


def status_message(optimizer, stats, step):
    """A basic console printout."""
    current_lr = f'{optimizer.param_groups[0]["lr"]:.4f}'

    def _maybe_print(key):
        return stats[key][-1] if len(stats[key]) > 0 else float("NaN")

    msg = f'Step: {step:<8}| lr: {current_lr} | Time: {stats["train_time"][-1]:4.2f}s |'
    msg += f'TRAIN loss {stats["train_loss"][-1]:7.4f} | TRAIN Acc: {stats["train_acc"][-1]:7.2%} |'
    for entry in stats.keys():
        if "valid" in entry and "cls" not in entry:
            if "loss" in entry:
                msg += f' {entry.split("_")[0]} VAL loss {_maybe_print(entry):7.4f} |'
            else:
                msg += f' {entry.split("_")[0]} VAL Acc {_maybe_print(entry):7.2%} |'
    return msg
