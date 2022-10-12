"""Train a classification model by with SGD. This variant is used for data aug experiments.

This CLI interface trains based on the hydra configuration received at startup."""

import torch
import hydra

import time
import logging

import dataaug

# import os

# os.environ["HYDRA_FULL_ERROR"] = "1"
log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="cfg", version_base="1.1")
def main_launcher(cfg):
    dataaug.utils.job_startup(main_process, cfg, log, "SGD computation")


def main_process(process_idx, local_group_size, cfg):
    local_time = time.time()
    setup = dataaug.utils.system_startup(process_idx, local_group_size, cfg)

    trainloader, validloaders = dataaug.data.construct_dataloader(cfg.data, cfg.impl, cfg.hyp, cfg.dryrun)
    model = dataaug.models.construct_model(cfg.model, cfg.data.channels, cfg.data.classes)
    model = dataaug.models.prepare_model(model, cfg, process_idx, setup)

    stats = dataaug.training.train_sgd(model, trainloader, validloaders, setup, cfg)
    stats["dataset_size"] += [len(trainloader.dataset)]
    if dataaug.utils.is_main_process():
        dataaug.utils.save_summary(cfg, stats, time.time() - local_time, exp="data_augs_")

    if cfg.impl.setup.dist:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main_launcher()
